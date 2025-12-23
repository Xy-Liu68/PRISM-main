# ./PRISM-main/train_evaluation.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from tqdm import tqdm
import argparse
import os
import time

from data_loader import QuadModalRNADataset, filter_none_collate_quad
from model import QuadModalBinder

def loop(model, dataloader, optimizer, criterion, device, is_train=True, current_epoch=0, total_epochs=0):
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0, [], []
    
    desc_prefix = f"Epoch {current_epoch}/{total_epochs}" if is_train else "Testing"
    
    pbar = tqdm(dataloader, desc=f"{desc_prefix} {'Training' if is_train else 'Evaluating'}", leave=False)
    
    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            if batch is None: continue
            
            rna_3d_data, rna_seq_data, mol_graph_data, smiles_data, labels = batch
            rna_3d_data, rna_seq_data, mol_graph_data, smiles_data, labels = rna_3d_data.to(device), rna_seq_data.to(device), mol_graph_data.to(device), smiles_data.to(device), labels.to(device)
            
            if is_train: optimizer.zero_grad()
            
            logits = model(rna_3d_data, rna_seq_data, mol_graph_data, smiles_data).squeeze(-1)
            loss = criterion(logits, labels)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).cpu().detach().numpy())

    if not all_labels: return {"loss": 0, "acc": 0, "f1": 0, "rocauc": 0, "prauc": 0, "pre": 0, "rec": 0}
    
    avg_loss = total_loss / len(all_labels)
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    
    return {
        "loss": avg_loss,
        "acc": accuracy_score(all_labels, all_preds),
        "pre": precision_score(all_labels, all_preds, zero_division=0),
        "rec": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "rocauc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        "prauc": average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }

def get_valid_indices(data_dir, all_indices, dataset_name):
    """Checks for successfully processed pairs."""
    valid_indices = []
    pbar = tqdm(all_indices, desc=f"Verifying {dataset_name} data", leave=True)
    for idx in pbar:
        pair_folder = os.path.join(data_dir, f'pair_{idx:05d}')
        mol_path = os.path.join(pair_folder, 'molecule_graph.npz')
        pdb_path = os.path.join(pair_folder, 'unrelaxed_model.pdb')
        if os.path.exists(mol_path) and os.path.exists(pdb_path):
            valid_indices.append(idx)
    return np.array(valid_indices)

def get_benchmark_indices(base_data_path, processed_benchmark_dir):
    """
    Determines index ranges for each benchmark dataset and filters for valid samples.
    """
    benchmark_files = {
        'NALDB': 'benchmark_NALDB.txt', 'NewPub': 'benchmark_NewPub.txt',
        'RBIND': 'benchmark_RBIND.txt', 'RSIM': 'benchmark_RSIM.txt',
        'SMMRNA': 'benchmark_SMMRNA.txt'
    }
    benchmark_indices = {}
    current_index = 0
    
    print("\n--- Determining benchmark dataset indices and verifying data ---")
    for name, filename in benchmark_files.items():
        path = os.path.join(base_data_path, filename)
        try:
            with open(path, 'r') as f: num_lines = sum(1 for line in f)
            indices_range = list(range(current_index, current_index + num_lines))
            valid_indices = get_valid_indices(processed_benchmark_dir, indices_range, name)
            benchmark_indices[name] = valid_indices
            print(f"  - {name}: Found {num_lines} total samples, {len(valid_indices)} successfully processed.")
            current_index += num_lines
        except FileNotFoundError:
            print(f"  [Warning] Benchmark file not found: {path}. Skipping.")
    return benchmark_indices

def main(config):
    log_dir = os.path.dirname(config['log_file'])
    os.makedirs(log_dir, exist_ok=True)
    final_model_path = os.path.join(log_dir, "model_evaluation.pth")
    
    with open(config['log_file'], 'w') as log_f:
        def log_message(message):
            print(message)
            log_f.write(message + '\n'); log_f.flush()

        device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
        log_message(f"--- Cold Start Experiment for Quad-Modal (PRISM) Model ---")
        log_message(f"Using device: {device}\nConfig: {config}\n")
        
        # 1. SETUP TRAINING DATA
        log_message("--- Loading and verifying training data ---")
        df_train = pd.read_csv(config['train_data_file'], sep='\t', header=None, names=['smiles', 'seq', 'ss', 'label'])
        train_indices_all = np.arange(len(df_train))
        train_indices_valid = get_valid_indices(config['train_processed_dir'], train_indices_all, "Training Set")
        log_message(f"Found {len(train_indices_valid)} valid training samples out of {len(train_indices_all)} total.")
        
        train_dataset = QuadModalRNADataset(config['train_processed_dir'], train_indices_valid, config['train_data_file'], augmentation=True)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])

        # 2. SETUP MODEL, LOSS, OPTIMIZER
        model = QuadModalBinder(config).to(device)
        train_labels = df_train.iloc[train_indices_valid]['label'].values
        pos_weight = torch.tensor(np.sum(train_labels == 0) / np.sum(train_labels == 1), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        log_message(f"Using weighted loss. Pos_weight: {pos_weight.item():.2f}")
        
        # 3. TRAINING LOOP
        log_message("\n--- Starting Model Training ---")
        start_time = time.time()
        for epoch in range(1, config['epochs'] + 1):
            train_metrics = loop(model, train_loader, optimizer, criterion, device, is_train=True, current_epoch=epoch, total_epochs=config['epochs'])
            log_message(f"Epoch {epoch}/{config['epochs']} | Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")
        
        training_duration = time.time() - start_time
        log_message(f"--- Training finished in {training_duration:.2f} seconds. ---")
        torch.save(model.state_dict(), final_model_path)
        log_message(f"Final trained model saved to {final_model_path}\n")

        # 4. SETUP AND RUN TESTING
        log_message("--- Preparing Benchmark Test Sets for Evaluation ---")
        benchmark_indices = get_benchmark_indices(config['benchmark_data_path'], config['benchmark_processed_dir'])
        df_benchmark = pd.read_csv(config['benchmark_merged_file'], sep='\t', header=None, names=['id', 'smiles', 'seq', 'ss', 'label'])
        
        log_message("\n--- Starting Evaluation on Benchmark Datasets ---")
        for test_name, test_indices in benchmark_indices.items():
            if len(test_indices) == 0:
                log_message(f"\n--- Skipping {test_name}: No valid samples found. ---")
                continue

            test_dataset = QuadModalRNADataset(
                root_dir=config['benchmark_processed_dir'], 
                pair_indices=test_indices, 
                smrtnet_data_file=config['benchmark_merged_file'],
                is_benchmark=True, 
                augmentation=False
            )
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])
            
            test_metrics = loop(model, test_loader, None, criterion, device, is_train=False)
            
            log_message(f"\n--- Test Results for: {test_name} ({len(test_indices)} samples) ---")
            log_message(f"  Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}, Precision: {test_metrics['pre']:.4f}, Recall: {test_metrics['rec']:.4f}")
            log_message(f"  F1: {test_metrics['f1']:.4f}, ROC-AUC: {test_metrics['rocauc']:.4f}, PR-AUC: {test_metrics['prauc']:.4f}")

        log_message("\n--- Cold Start Experiment Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Test Quad-Modal SMRTBinder for Cold Start Generalization')
    parser.add_argument('--train_data_file', type=str, default='./PRISM-main/data/SMRTnet_data.txt')
    parser.add_argument('--train_processed_dir', type=str, default='./PRISM-main/3D_processed_output/smrtnet_processed_output')
    parser.add_argument('--benchmark_data_path', type=str, default='./PRISM-main/data/', help="Path to directory with original benchmark .txt files")
    parser.add_argument('--benchmark_merged_file', type=str, default='./PRISM-main/data/benchmark_all.txt')
    parser.add_argument('--benchmark_processed_dir', type=str, default='./PRISM-main/3D_processed_output/all_benchmark_processed_output')
    
    parser.add_argument('--log_file', type=str, default='./PRISM-main/results/evaluation_log.txt')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--mol_feat_dim', type=int, default=41)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    main(vars(args))

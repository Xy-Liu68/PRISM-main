# ./PRISM-main/train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os
import time
from data_loader import QuadModalRNADataset, filter_none_collate_quad
from model import QuadModalBinder

def loop(model, dataloader, optimizer, criterion, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0, [], []
    desc = "Training" if is_train else "Evaluating"
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch_globals['current']}/{epoch_globals['total']}] {desc}", leave=False)
    
    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            if batch is None: continue
            
            rna_3d_data, rna_seq_data, mol_graph_data, smiles_data, labels = batch
            
            rna_3d_data = rna_3d_data.to(device)
            rna_seq_data = rna_seq_data.to(device)
            mol_graph_data = mol_graph_data.to(device)
            smiles_data = smiles_data.to(device)
            labels = labels.to(device)
            
            if is_train: optimizer.zero_grad()
            
            logits = model(rna_3d_data, rna_seq_data, mol_graph_data, smiles_data).squeeze(-1)
            loss = criterion(logits, labels)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).cpu().detach().numpy())

            if is_train:
                running_labels = np.array(all_labels)
                running_preds = (np.array(all_probs) > 0.5).astype(int)

                if len(np.unique(running_labels)) > 1:
                    live_metrics = {
                        'los': loss.item(),
                        'acc': accuracy_score(running_labels, running_preds),
                        'pre': precision_score(running_labels, running_preds, zero_division=0),
                        'rec': recall_score(running_labels, running_preds, zero_division=0),
                        'f1': f1_score(running_labels, running_preds, zero_division=0),
                        'auc': roc_auc_score(running_labels, np.array(all_probs)),
                        'prc': average_precision_score(running_labels, np.array(all_probs))
                    }
                else:
                    live_metrics = {
                        'los': loss.item(),
                        'acc': accuracy_score(running_labels, running_preds),
                        'pre': precision_score(running_labels, running_preds, zero_division=0),
                        'rec': recall_score(running_labels, running_preds, zero_division=0),
                        'f1': f1_score(running_labels, running_preds, zero_division=0)
                    }
                pbar.set_postfix({k: f"{v:.3f}" for k, v in live_metrics.items()})
            else:
                 pbar.set_postfix({'val_loss': loss.item()})
            
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
    valid_indices = []
    pbar = tqdm(all_indices, desc=f"Verifying {dataset_name} data", leave=False)
    for idx in pbar:
        pair_folder = os.path.join(data_dir, f'pair_{idx:05d}')
        mol_path = os.path.join(pair_folder, 'molecule_graph.npz')
        pdb_path = os.path.join(pair_folder, 'unrelaxed_model.pdb')
        if os.path.exists(mol_path) and os.path.exists(pdb_path):
            valid_indices.append(idx)
    return np.array(valid_indices)

def main(config):
    global epoch_globals
    epoch_globals = {'current': 0, 'total': config['epochs']}    
    log_dir = os.path.dirname(config['log_file'])
    os.makedirs(log_dir, exist_ok=True)
    
    with open(config['log_file'], 'w') as log_f:
        def log_message(message):
            print(message)
            log_f.write(message + '\n'); log_f.flush()
        device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
        log_message(f"Using device: {device}\nConfig: {config}\n")
        
        log_message("--- Loading and verifying the main dataset ---")
        df_full = pd.read_csv(config['data_file'], sep='\t', header=None, names=['smiles', 'seq', 'ss', 'label'])
        
        all_indices = np.arange(len(df_full))
        valid_indices = get_valid_indices(config['data_dir'], all_indices, "Full Dataset")
        
        labels_valid = df_full.iloc[valid_indices]['label'].values
        log_message(f"Total valid samples found: {len(valid_indices)}")
        
        all_folds_test_metrics = []
        log_message(f"\n--- Starting {config['kfold']}-Fold Training Process (with 80/10/10 split each fold) ---")
        
        for fold in range(config['kfold']):
            start_time_fold = time.time()
            log_message(f"\n{'='*25} FOLD {fold + 1}/{config['kfold']} {'='*25}")
            
            current_seed = config['seed'] + fold
            
            train_indices, temp_test_indices, _, temp_test_labels = train_test_split(
                valid_indices, labels_valid,
                test_size=(config['val_size'] + config['test_size']),
                stratify=labels_valid, random_state=current_seed
            )
            
            val_indices, test_indices, _, _ = train_test_split(
                temp_test_indices, temp_test_labels,
                test_size=0.5, stratify=temp_test_labels, random_state=current_seed
            )
            
            log_message(f"  Random seed for this fold's split: {current_seed}")
            log_message(f"  Train: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_indices)}")
            
            train_dataset = QuadModalRNADataset(config['data_dir'], train_indices, config['data_file'], augmentation=True)
            val_dataset = QuadModalRNADataset(config['data_dir'], val_indices, config['data_file'], augmentation=False)
            test_dataset = QuadModalRNADataset(config['data_dir'], test_indices, config['data_file'], augmentation=False)
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])
            model = QuadModalBinder(config).to(device)
            train_labels = df_full.loc[train_indices, 'label'].values
            neg_num, pos_num = np.sum(train_labels == 0), np.sum(train_labels == 1)
            pos_weight = torch.tensor(neg_num / pos_num if pos_num > 0 else 1.0, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            log_message(f"  Pos_weight for this fold: {pos_weight.item():.2f}")
            optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=False)
            
            best_model_path_fold = os.path.join(log_dir, f"model_fold_{fold+1}_best.pth")
            
            best_f1_fold = 0.0
            epochs_no_improve = 0
            for epoch in range(1, config['epochs'] + 1):
                epoch_globals['current'] = epoch
                
                train_metrics = loop(model, train_loader, optimizer, criterion, device, is_train=True)
                val_metrics = loop(model, val_loader, None, criterion, device, is_train=False)
                
                log_msg = (f"Valid at Epoch: {epoch}, "
                           f"train_loss: {train_metrics['loss']:.3f} "
                           f"valid_loss: {val_metrics['loss']:.3f} "
                           f"valid_acc: {val_metrics['acc']:.3f} "
                           f"valid_pre: {val_metrics['pre']:.3f} "
                           f"valid_rec: {val_metrics['rec']:.3f} "
                           f"valid_auc: {val_metrics['rocauc']:.3f} "
                           f"valid prc: {val_metrics['prauc']:.3f}")
                log_message(log_msg)
                scheduler.step(val_metrics['f1'])
                
                if val_metrics['f1'] > best_f1_fold:
                    best_f1_fold = val_metrics['f1']
                    torch.save(model.state_dict(), best_model_path_fold)
                    log_message(f"   ---> Best model saved with F1: {best_f1_fold:.4f}") 
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= config['patience']:
                    log_message(f"  --- Early stopping triggered in Fold {fold+1} at epoch {epoch}. ---") 
                    break
            log_message(f"--- Fold {fold+1} training finished. Loading best model for evaluation on the TEST set. ---")
            model.load_state_dict(torch.load(best_model_path_fold))
            final_test_metrics_fold = loop(model, test_loader, None, criterion, device, is_train=False)
            all_folds_test_metrics.append(final_test_metrics_fold)            
            fold_duration = time.time() - start_time_fold            
            final_log_msg = (f"\n--- Final TEST Results for Fold {fold+1} (Best Model) ---\n"
                             f"  Loss: {final_test_metrics_fold['loss']:.4f}, Acc: {final_test_metrics_fold['acc']:.4f}, "
                             f"Precision: {final_test_metrics_fold['pre']:.4f}, Recall: {final_test_metrics_fold['rec']:.4f}\n"
                             f"  F1: {final_test_metrics_fold['f1']:.4f}, ROC-AUC: {final_test_metrics_fold['rocauc']:.4f}, "
                             f"PR-AUC: {final_test_metrics_fold['prauc']:.4f}\n"
                             f"--- Fold {fold+1} completed in {fold_duration:.2f} seconds. ---")
            log_message(final_log_msg)
        
        log_message(f"\n\n{'='*20} 5-FOLD SUMMARY (based on TEST set performance) {'='*20}")
        
        metrics_df = pd.DataFrame(all_folds_test_metrics)
        avg_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()
        
        log_message("--- Metrics for each fold on its respective TEST set ---")
        log_f.write(metrics_df.to_string() + '\n')
        
        log_message("\n--- Average Performance (Mean ± Std Dev) ---")  
        for metric in avg_metrics.index:
            log_message(f"  {metric.upper():<10}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Quad-Modal Model with 5-Fold CV and Detailed Logging')
    
    parser.add_argument('--data_file', type=str, default='./PRISM-main/data/SMRTnet_data.txt')
    parser.add_argument('--data_dir', type=str, default='./PRISM-main/3D_processed_output/smrtnet_processed_output')
    parser.add_argument('--log_file', type=str, default='./PRISM-main/results/training_log.txt')
    
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--mol_feat_dim', type=int, default=41)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10) 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("The sum of val_size and test_size must be less than 1.0")
        
    main(vars(args))
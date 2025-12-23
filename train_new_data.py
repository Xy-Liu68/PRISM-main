import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os
from model import QuadModalBinder

from data_loader import parse_pdb_to_graph_denoised, pad_sequence, filter_none_collate_quad
from data_loader import MAX_SEQ_LEN, MAX_SMILES_LEN, RNA_SEQ_CHAR_TO_INT, SMILES_CHAR_TO_INT


class CustomRNADataset(Dataset):
    def __init__(self, root_dir, pair_indices, data_file, augmentation=False):
        self.root_dir = root_dir
        self.pair_indices = pair_indices
        self.augmentation = augmentation
        self.df = pd.read_csv(data_file, sep='\t', header=None, names=['smiles', 'seq', 'label'])
        
    def __len__(self):
        return len(self.pair_indices)
        
    def __getitem__(self, idx):
        try:
            pair_idx = self.pair_indices[idx]
            row = self.df.iloc[pair_idx]
            pair_folder = os.path.join(self.root_dir, f'pair_{pair_idx:05d}')
            
            mol_path = os.path.join(pair_folder, 'molecule_graph.npz')
            pdb_path = os.path.join(pair_folder, 'unrelaxed_model.pdb')
            
            if not all(os.path.exists(p) for p in [mol_path, pdb_path]): return None

            with np.load(mol_path) as mol_data_np:
                mol_graph = Data(x=torch.tensor(mol_data_np['node_features'], dtype=torch.float32), 
                                 edge_index=torch.tensor(mol_data_np['edge_index'], dtype=torch.long))

            rna_3d_graph = parse_pdb_to_graph_denoised(pdb_path)
            if rna_3d_graph is None: return None
            
            if self.augmentation:
                rna_3d_graph.pos += torch.randn_like(rna_3d_graph.pos) * 0.1

            rna_seq_padded = pad_sequence(row['seq'], MAX_SEQ_LEN, RNA_SEQ_CHAR_TO_INT)
            smiles_padded = pad_sequence(row['smiles'], MAX_SMILES_LEN, SMILES_CHAR_TO_INT)
            label = float(row['label'])
            
            return rna_3d_graph, rna_seq_padded, mol_graph, smiles_padded, torch.tensor(label, dtype=torch.float32)
        
        except Exception: return None

from torch_geometric.data import Data

def loop(model, dataloader, optimizer, criterion, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0, [], []
    
    pbar = tqdm(dataloader, desc=f"{'Train' if is_train else 'Eval'}", leave=False, disable=not is_train)
    
    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
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
                if pbar: pbar.update(1)
                
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).cpu().detach().numpy())
            
    if not all_labels: return {"loss": 0, "acc": 0, "f1": 0, "rocauc": 0, "pre": 0, "rec": 0}
    
    avg_loss = total_loss / len(all_labels)
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return {
        "loss": avg_loss,
        "acc": accuracy_score(all_labels, all_preds),
        "pre": precision_score(all_labels, all_preds, zero_division=0),
        "rec": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "rocauc": auc
    }

def get_valid_indices(data_dir, all_indices):
    valid_indices = []
    for idx in all_indices:
        pair_folder = os.path.join(data_dir, f'pair_{idx:05d}')
        if os.path.exists(os.path.join(pair_folder, 'molecule_graph.npz')) and \
           os.path.exists(os.path.join(pair_folder, 'unrelaxed_model.pdb')):
            valid_indices.append(idx)
    return np.array(valid_indices)

def main(config):
    os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Training on Custom 3-Column Dataset ---")
    print(f"Data: {config['data_file']}")
    print(f"3D Dir: {config['data_dir']}")

    try:
        df = pd.read_csv(config['data_file'], sep='\t', header=None, names=['smiles', 'seq', 'label'])
    except Exception as e:
        print(f"Error reading data file. Ensure it has 3 columns (SMILES, SEQ, LABEL). {e}")
        return

    all_indices = np.arange(len(df))
    valid_indices = get_valid_indices(config['data_dir'], all_indices)
    
    print(f"Total: {len(df)}. Valid: {len(valid_indices)}")
    if len(valid_indices) == 0:
        print("No valid processed data found. Run process_new_data.py first.")
        return

    labels_valid = df.iloc[valid_indices]['label'].values

    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        valid_indices, labels_valid, test_size=0.2, stratify=labels_valid, random_state=config['seed']
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=config['seed']
    )

    train_set = CustomRNADataset(config['data_dir'], train_idx, config['data_file'], augmentation=True)
    val_set = CustomRNADataset(config['data_dir'], val_idx, config['data_file'], augmentation=False)
    test_set = CustomRNADataset(config['data_dir'], test_idx, config['data_file'], augmentation=False)
    
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, collate_fn=filter_none_collate_quad, num_workers=config['num_workers'])

    model = QuadModalBinder(config).to(device)
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    pos_count = np.sum(train_labels==1)
    neg_count = np.sum(train_labels==0)
    pos_weight = torch.tensor(neg_count/pos_count if pos_count > 0 else 1.0, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_f1 = 0
    save_path = os.path.join(os.path.dirname(config['log_file']), 'best_custom_model.pth')

    print("\nStarting Training...")
    for epoch in range(1, config['epochs'] + 1):
        train_met = loop(model, train_loader, optimizer, criterion, device, is_train=True)
        val_met = loop(model, val_loader, None, criterion, device, is_train=False)
        
        scheduler.step(val_met['f1'])
        
        if val_met['f1'] > best_f1:
            best_f1 = val_met['f1']
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch}: New Best F1: {best_f1:.4f} (Saved)")
        else:
            print(f"Epoch {epoch}: Train Loss: {train_met['loss']:.3f}, Val F1: {val_met['f1']:.3f}")

    # 5. Final Test
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(save_path))
    test_met = loop(model, test_loader, None, criterion, device, is_train=False)
    
    print("-" * 30)
    print(f"Test Accuracy:  {test_met['acc']:.4f}")
    print(f"Test F1 Score:  {test_met['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_met['rocauc']:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Path to your .txt data file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the processed 3D output directory')
    parser.add_argument('--log_file', type=str, default='./PRISM-main/results/custom_training_log.txt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--mol_feat_dim', type=int, default=41)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(vars(args))

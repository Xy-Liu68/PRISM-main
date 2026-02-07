# ./PRISM-main/train_evaluation_robin.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from tqdm import tqdm
import argparse
import os
import sys
import datetime
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from model import QuadModalBinder
from data_loader import (
    filter_none_collate_quad, 
    parse_pdb_to_graph_denoised, 
    pad_sequence, 
    MAX_SEQ_LEN, 
    MAX_SMILES_LEN, 
    RNA_SEQ_CHAR_TO_INT, 
    SMILES_CHAR_TO_INT
)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'Mg', 'Ca', 'Na', 'K', 'Li', 'Al', 'Ag', 'other']
    feat_symbol = one_of_k_encoding(atom.GetSymbol(), symbols)
    feat_degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    feat_valence = one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    hybridization = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D]
    feat_hybrid = one_of_k_encoding(atom.GetHybridization(), hybridization)
    feat_aromatic = [1 if atom.GetIsAromatic() else 0]
    
    results = feat_symbol + feat_degree + feat_valence + feat_hybrid + feat_aromatic
    if len(results) != 41:
        if len(results) > 41: results = results[:41]
        else: results = results + [0] * (41 - len(results))
    return np.array(results).astype(np.float32)

def smiles_to_graph_41dim(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_feats.append(atom_features(atom))
        x = torch.tensor(np.array(atom_feats), dtype=torch.float32)
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append((i, j)); edges.append((j, i))
        if len(edges) == 0: edge_index = torch.empty((2, 0), dtype=torch.long)
        else: edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    except: return None

def build_sequence_to_pdb_map(pdb_root_dir):
    print(f"[Init] Scanning 3D structures in {pdb_root_dir}...")
    seq_map = {}
    if not os.path.exists(pdb_root_dir): return seq_map
    subdirs = [d for d in os.listdir(pdb_root_dir) if os.path.isdir(os.path.join(pdb_root_dir, d))]
    for folder_name in subdirs:
        folder_path = os.path.join(pdb_root_dir, folder_name)
        pdb_path = os.path.join(folder_path, "relaxed_1000_model.pdb")
        if not os.path.exists(pdb_path): pdb_path = os.path.join(folder_path, "unrelaxed_model.pdb")
        if not os.path.exists(pdb_path): continue
        try:
            ss_path = os.path.join(folder_path, "ss.ct")
            if os.path.exists(ss_path):
                with open(ss_path, 'r') as f:
                    lines = f.readlines()
                    seq_chars = [line.strip().split()[1] for line in lines[1:] if len(line.strip().split()) >= 2]
                    extracted_seq = "".join(seq_chars).upper().replace("T", "U")
                    if extracted_seq not in seq_map: seq_map[extracted_seq] = pdb_path
        except: pass
    print(f"[Init] Mapped {len(seq_map)} unique RNA sequences to PDB files.")
    return seq_map

class RobinTestDataset(Dataset):
    def __init__(self, csv_path, seq_to_pdb_map):
        self.data = pd.read_csv(csv_path)
        if 'rna' in self.data.columns: self.data = self.data.rename(columns={'rna': 'seq', 'ligand': 'smiles'})
        self.seq_to_pdb_map = seq_to_pdb_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            rna_seq = row['seq'].strip().upper().replace("T", "U")
            smiles = row['smiles']
            label = float(row['label'])

            pdb_path = self.seq_to_pdb_map.get(rna_seq)
            if not pdb_path:

                for k, v in self.seq_to_pdb_map.items():
                    if rna_seq in k or k in rna_seq: pdb_path = v; break
            if not pdb_path: return None

            rna_3d_graph = parse_pdb_to_graph_denoised(pdb_path)
            if rna_3d_graph is None: return None
            
            rna_seq_padded = pad_sequence(rna_seq, MAX_SEQ_LEN, RNA_SEQ_CHAR_TO_INT)
            smiles_padded = pad_sequence(smiles, MAX_SMILES_LEN, SMILES_CHAR_TO_INT)
            mol_graph = smiles_to_graph_41dim(smiles)
            if mol_graph is None: return None

            return rna_3d_graph, rna_seq_padded, mol_graph, smiles_padded, torch.tensor(label, dtype=torch.float32)
        except: return None

def evaluate_robin(model, dataloader, device, log_file):
    model.eval()
    total_loss = 0
    all_labels, all_probs = [], []
    criterion = nn.BCEWithLogitsLoss()

    with open(log_file, 'a') as f:
        f.write(f"\n[{datetime.datetime.now()}] Starting Evaluation on Robin Dataset...\n")
        print(">>> Starting Inference on Robin...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Robin", leave=True):
                if batch is None: continue
                rna_3d_data, rna_seq_data, mol_graph_data, smiles_data, labels = batch
                rna_3d_data, rna_seq_data, mol_graph_data, smiles_data, labels = \
                    rna_3d_data.to(device), rna_seq_data.to(device), mol_graph_data.to(device), smiles_data.to(device), labels.to(device)

                logits = model(rna_3d_data, rna_seq_data, mol_graph_data, smiles_data).squeeze(-1)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try: roc_auc = roc_auc_score(all_labels, all_probs)
    except: roc_auc = 0.5
    try: prauc = average_precision_score(all_labels, all_probs)
    except: prauc = 0.5
    
    avg_loss = total_loss / len(dataloader)

    result_msg = (
        f"\n{'='*30}\n"
        f"*** ROBIN DATASET GENERALIZATION RESULTS ***\n"
        f"{'='*30}\n"
        f"Samples Evaluated: {len(all_labels)}\n"
        f"Loss     : {avg_loss:.4f}\n"
        f"ROC-AUC  : {roc_auc:.4f}\n"
        f"PR-AUC   : {prauc:.4f}\n"
        f"F1 Score : {f1:.4f}\n"
        f"Accuracy : {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall   : {rec:.4f}\n"
        f"{'='*30}\n"
    )
    
    print(result_msg)
    with open(log_file, 'a') as f:
        f.write(result_msg)

def main():
    parser = argparse.ArgumentParser(description='Evaluate PRISM-trained Model on Robin HTS Dataset')
    parser.add_argument('--robin_csv', type=str, default='./PRISM-main/data/Robin.csv')
    parser.add_argument('--pdb_root_dir', type=str, default='./PRISM-main/data/3d')
    parser.add_argument('--model_path', type=str, default='./PRISM-main/results/quad_model_cold_start_final.pth') 
    parser.add_argument('--output_log', type=str, default='./PRISM-main/results/robin_generalization_test.txt')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--mol_feat_dim', type=int, default=41) 
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)

    print(f"Loading pre-trained model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}. Please run train_quad_cold_start.py first.")
        return

    config = {'hidden_dim': args.hidden_dim, 'mol_feat_dim': args.mol_feat_dim}
    model = QuadModalBinder(config).to(args.device)

    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    print("Building RNA PDB Map...")
    seq_map = build_sequence_to_pdb_map(args.pdb_root_dir)
    
    print("Initializing Robin Dataset...")
    robin_dataset = RobinTestDataset(args.robin_csv, seq_map)
    print(f"Robin Dataset Size: {len(robin_dataset)}")
    
    robin_loader = DataLoader(robin_dataset, batch_size=args.batch_size, shuffle=False, 
                              collate_fn=filter_none_collate_quad, num_workers=args.num_workers)

    evaluate_robin(model, robin_loader, args.device, args.output_log)

if __name__ == '__main__':
    main()
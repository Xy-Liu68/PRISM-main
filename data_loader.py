# ./PRISM-main/data_loader.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from scipy.spatial.distance import cdist

MAX_SEQ_LEN = 256
MAX_SMILES_LEN = 150
RNA_SEQ_CHAR_TO_INT = {'A': 1, 'U': 2, 'C': 3, 'G': 4, 'N': 5}
RNA_SEQ_VOCAB_SIZE = len(RNA_SEQ_CHAR_TO_INT) + 1
SMILES_CHAR_TO_INT = {
    '#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9, '4': 10, 
    '5': 11, '6': 12, '7': 13, '8': 14, '=': 15, 'B': 16, 'C': 17, 'F': 18, 'H': 19, 
    'I': 20, 'N': 21, 'O': 22, 'P': 23, 'S': 24, '[': 25, '\\': 26, ']': 27, 'c': 28, 
    'i': 29, 'l': 30, 'n': 31, 'o': 32, 'r': 33, 's': 34
}
SMILES_VOCAB_SIZE = len(SMILES_CHAR_TO_INT) + 1

BASE_TYPE_MAP = {'A': 0, 'U': 1, 'C': 2, 'G': 3, 'N': 4} 

def pad_sequence(seq, max_len, char_to_int_map):
    padd_val = 0
    tokenized = [char_to_int_map.get(c, padd_val) for c in seq]
    padded_len = max_len - len(tokenized)
    return torch.tensor(tokenized[:max_len] + [padd_val] * padded_len, dtype=torch.long)

def parse_pdb_to_graph_denoised(pdb_path, max_nodes=2048, dist_thresh=16.0):
    try:
        nodes_info = {}
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and 'C4\'' in line:
                    res_id = int(line[22:26].strip())
                    if res_id not in nodes_info:
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        base = line[17:20].strip()[0]
                        b_factor = float(line[60:66].strip())
                        nodes_info[res_id] = {'pos': [x, y, z], 'base': base, 'b_factor': b_factor}
        if not nodes_info or len(nodes_info) > max_nodes: return None
        sorted_res_ids = sorted(nodes_info.keys())
        positions = np.array([nodes_info[rid]['pos'] for rid in sorted_res_ids])
        b_factors = np.array([nodes_info[rid]['b_factor'] for rid in sorted_res_ids])
        base_indices = [BASE_TYPE_MAP.get(nodes_info[rid]['base'], BASE_TYPE_MAP['N']) for rid in sorted_res_ids]
        base_one_hot = np.eye(len(BASE_TYPE_MAP))[base_indices]
        normalized_b_factor = (b_factors - b_factors.mean()) / (b_factors.std() + 1e-6)
        node_feats = np.concatenate([base_one_hot, normalized_b_factor[:, np.newaxis]], axis=1, dtype=np.float32)
        dist_matrix = cdist(positions, positions)
        spatial_adj = (dist_matrix > 0) & (dist_matrix < dist_thresh)
        spatial_edges = np.array(np.where(spatial_adj))
        res_id_map = {rid: i for i, rid in enumerate(sorted_res_ids)}
        backbone_edges_set = set()
        for i in range(len(sorted_res_ids) - 1):
            if sorted_res_ids[i+1] == sorted_res_ids[i] + 1:
                idx1, idx2 = res_id_map[sorted_res_ids[i]], res_id_map[sorted_res_ids[i+1]]
                backbone_edges_set.add((idx1, idx2)); backbone_edges_set.add((idx2, idx1))
        backbone_edges = np.array(list(backbone_edges_set)).T if backbone_edges_set else np.empty((2, 0), dtype=np.int64)
        edge_index = np.concatenate([spatial_edges, backbone_edges], axis=1)
        edge_attr_spatial = np.zeros((spatial_edges.shape[1], 2), dtype=np.float32)
        edge_attr_spatial[:, 0] = 1.0 / (dist_matrix[spatial_edges[0], spatial_edges[1]] + 1e-6)
        edge_attr_backbone = np.zeros((backbone_edges.shape[1], 2), dtype=np.float32)
        if backbone_edges.shape[1] > 0:
            edge_attr_backbone[:, 0] = 1.0 / (dist_matrix[backbone_edges[0], backbone_edges[1]] + 1e-6)
            edge_attr_backbone[:, 1] = 1.0
        edge_attr = np.concatenate([edge_attr_spatial, edge_attr_backbone], axis=0)
        return Data(x=torch.tensor(node_feats, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    pos=torch.tensor(positions, dtype=torch.float32),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float32))
    except Exception: return None

class QuadModalRNADataset(Dataset):
    def __init__(self, root_dir, pair_indices, smrtnet_data_file, is_benchmark=False, augmentation=False):
        self.root_dir = root_dir
        self.pair_indices = pair_indices
        self.augmentation = augmentation
        df_cols = ['id', 'smiles', 'seq', 'ss', 'label'] if is_benchmark else ['smiles', 'seq', 'ss', 'label']
        self.df = pd.read_csv(smrtnet_data_file, sep='\t', header=None, names=df_cols)
        
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

def filter_none_collate_quad(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    rna_3d_graphs, rna_seqs, mol_graphs, smiles_seqs, labels = zip(*batch)    
    rna_3d_batch = Batch.from_data_list(list(rna_3d_graphs))
    mol_batch = Batch.from_data_list(list(mol_graphs))
    rna_seq_batch = default_collate(rna_seqs)
    smiles_batch = default_collate(smiles_seqs)
    labels = default_collate(labels)
    
    return rna_3d_batch, rna_seq_batch, mol_batch, smiles_batch, labels

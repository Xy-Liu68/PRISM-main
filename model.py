# ./PRISM-main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse_Network
from data_loader import BASE_TYPE_MAP, RNA_SEQ_VOCAB_SIZE, SMILES_VOCAB_SIZE

class RNA3DEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim, edge_feat_dim, n_layers=4):
        super().__init__()
        self.egnn = EGNN_Sparse_Network(
            n_layers=n_layers,
            pos_dim=3,
            feats_dim=input_feat_dim,
            edge_attr_dim=edge_feat_dim,
            m_dim=hidden_dim,
            update_coors=False,
            norm_feats=True
        )
        self.projection = nn.Linear(input_feat_dim, hidden_dim)

    def forward(self, rna_data):
        combined_input = torch.cat([rna_data.x, rna_data.pos], dim=-1)
        output_combined = self.egnn(combined_input, rna_data.edge_index, edge_attr=rna_data.edge_attr, batch=rna_data.batch)
        updated_feats = output_combined[:, :rna_data.x.size(1)]
        graph_embedding = global_mean_pool(updated_feats, rna_data.batch)
        return self.projection(graph_embedding)

class MoleculeGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.gcn_layers = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(n_layers - 1): self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, mol_data):
        x = mol_data.x
        for layer in self.gcn_layers: x = F.relu(layer(x, mol_data.edge_index))
        return self.projection(global_mean_pool(x, mol_data.batch))

class RNASeqEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim=128, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, seq_data):
        embedded = self.embedding(seq_data) # (B, L) -> (B, L, D)
        embedded = embedded.permute(0, 2, 1) # (B, D, L) for Conv1d
        cnn_out = self.cnn(embedded)
        pooled = self.pool(cnn_out).squeeze(-1) # (B, H)
        return self.projection(pooled)

class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim=128, kernel_size=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.cnn = nn.Sequential(

            nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, smiles_data):
        embedded = self.embedding(smiles_data)
        embedded = embedded.permute(0, 2, 1)
        cnn_out = self.cnn(embedded)
        pooled = self.pool(cnn_out).squeeze(-1)
        return self.projection(pooled)

class GatedFusionClassifier(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, feat_3d, feat_seq, feat_mol_graph, feat_smiles):
        combined_for_gate = torch.cat([feat_3d, feat_seq, feat_mol_graph, feat_smiles], dim=1)
        gates = self.gate_network(combined_for_gate)
        
        g_3d, g_seq, g_mol, g_smiles = gates[:, 0].unsqueeze(1), gates[:, 1].unsqueeze(1), gates[:, 2].unsqueeze(1), gates[:, 3].unsqueeze(1)
        
        fused_feat = g_3d * feat_3d + g_seq * feat_seq + g_mol * feat_mol_graph + g_smiles * feat_smiles
        return self.classifier(fused_feat)

class QuadModalBinder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['hidden_dim']

        # 1. RNA 3D Encoder
        self.rna_3d_encoder = RNA3DEncoder(
            input_feat_dim=len(BASE_TYPE_MAP) + 1,
            hidden_dim=hidden_dim,
            edge_feat_dim=2 
        )
        # 2. RNA Sequence Encoder
        self.rna_seq_encoder = RNASeqEncoder(RNA_SEQ_VOCAB_SIZE, hidden_dim)
        
        # 3. Molecule Graph Encoder
        self.mol_graph_encoder = MoleculeGraphEncoder(config['mol_feat_dim'], hidden_dim)

        # 4. SMILES Sequence Encoder
        self.smiles_encoder = SmilesEncoder(SMILES_VOCAB_SIZE, hidden_dim)

        # Normalization layers for each modality's features
        self.norm_rna_3d = nn.LayerNorm(hidden_dim)
        self.norm_rna_seq = nn.LayerNorm(hidden_dim)
        self.norm_mol_graph = nn.LayerNorm(hidden_dim)
        self.norm_smiles = nn.LayerNorm(hidden_dim)
        
        self.fusion_classifier = GatedFusionClassifier(hidden_dim)

    def forward(self, rna_3d_data, rna_seq_data, mol_graph_data, smiles_data):
        # Feature extraction from each encoder
        rna_3d_feat = self.rna_3d_encoder(rna_3d_data)
        rna_seq_feat = self.rna_seq_encoder(rna_seq_data)
        mol_graph_feat = self.mol_graph_encoder(mol_graph_data)
        smiles_feat = self.smiles_encoder(smiles_data)

        rna_3d_feat = self.norm_rna_3d(rna_3d_feat)
        rna_seq_feat = self.norm_rna_seq(rna_seq_feat)
        mol_graph_feat = self.norm_mol_graph(mol_graph_feat)
        smiles_feat = self.norm_smiles(smiles_feat)

        return self.fusion_classifier(rna_3d_feat, rna_seq_feat, mol_graph_feat, smiles_feat)

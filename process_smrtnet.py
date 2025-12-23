import os
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import sys

SMRTNET_DATA_FILE = './PRISM-main/data/SMRTnet_data.txt'
RHOFOLD_SCRIPT_PATH = './PRISM-main/Reference_module/RhoFold-main/inference.py'
RHOFOLD_ROOT_DIR = './PRISM-main/Reference_module/RhoFold-main'
RHOFOLD_CKPT_PATH = './PRISM-main/Reference_module/RhoFold-main/pretrained/rhofold_pretrained_params.pt'
OUTPUT_DIR = './PRISM-main/3D_processed_output/smrtnet_processed_output'

DEVICE = 'cuda:0' 

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(),
                                     ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other'])
    encoding += [atom.GetIsAromatic()]
    return np.array(encoding, dtype=np.float32)

def smiles_to_2d_graph(smi):
    mol = MolFromSmiles(smi)
    if mol is None: return None
    nodes_feat = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
    edge_index_np = np.array(edge_index, dtype=np.int64).T
    return {'node_features': np.array(nodes_feat), 'edge_index': edge_index_np}

def run_rhofold_for_sequence(sequence, fasta_path, output_dir):

    header = ">temp_rna_sequence"
    with open(fasta_path, 'w') as f:
        f.write(f"{header}\n{sequence}\n")
    command = [
        'python', RHOFOLD_SCRIPT_PATH,
        '--input_fas', fasta_path,
        '--output_dir', output_dir,
        '--ckpt', RHOFOLD_CKPT_PATH,
        '--single_seq_pred', 'True',
        '--device', DEVICE,
        '--relax_steps', '0'
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=RHOFOLD_ROOT_DIR
        )
    except subprocess.CalledProcessError as e:
        tqdm.write(f"  [ERROR] RhoFold+ failed for sequence: {sequence[:30]}...")
        return None
    except FileNotFoundError:
        tqdm.write(f"[FATAL ERROR] Cannot find 'python' or script '{RHOFOLD_SCRIPT_PATH}'.")
        return None
    pdb_path = os.path.join(output_dir, 'unrelaxed_model.pdb')
    if os.path.exists(pdb_path):
        return pdb_path
    else:
        tqdm.write(f"  [ERROR] RhoFold+ ran but did not produce the expected PDB file for {os.path.basename(output_dir)}.")
        return None

def main():
    print("Starting SMRTnet data processing...")
    if not os.path.exists(SMRTNET_DATA_FILE):
        print(f"[FATAL ERROR] Input data file not found: {SMRTNET_DATA_FILE}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        df = pd.read_csv(SMRTNET_DATA_FILE, sep='\t', header=None, names=['smiles', 'rna_sequence', 'dot_bracket', 'label'])
    except FileNotFoundError:
        print(f"[FATAL ERROR] SMRTnet data file not found at: {SMRTNET_DATA_FILE}")
        return
        
    print(f"Found {len(df)} pairs to process.")
    success_count, fail_count = 0, 0
    progress_bar = tqdm(df.iterrows(), total=df.shape[0], desc="Initializing...", file=sys.stdout)
    for index, row in progress_bar:
        pair_id = f"pair_{index:05d}"
        progress_bar.set_description(f"Processing {pair_id}")        
        pair_output_dir = os.path.join(OUTPUT_DIR, pair_id)
        os.makedirs(pair_output_dir, exist_ok=True)
        mol_graph = smiles_to_2d_graph(row['smiles'])
        if mol_graph is None:
            tqdm.write(f"[ERROR] Failed to process SMILES for {pair_id}: {row['smiles']}")
            fail_count += 1
            with open(os.path.join(pair_output_dir, "error.log"), "w") as f:
                f.write("SMILES processing failed.")
            continue
        mol_graph_path = os.path.join(pair_output_dir, 'molecule_graph.npz')
        np.savez(mol_graph_path, node_features=mol_graph['node_features'], edge_index=mol_graph['edge_index'])
        
        rna_seq_cleaned = row['rna_sequence'].upper().replace('T', 'U')
        temp_fasta_path = os.path.join(pair_output_dir, 'temp_rna.fasta')
        
        pdb_file = run_rhofold_for_sequence(rna_seq_cleaned, temp_fasta_path, pair_output_dir)
        if pdb_file:

            if success_count % 10 == 0:
                 tqdm.write(f"Successfully generated: {pair_id} (Total success: {success_count + 1})")
            success_count += 1
        else:
            fail_count += 1

            with open(os.path.join(pair_output_dir, "error.log"), "a") as f:
                f.write("\nRhoFold+ failed.")
            continue
        
        if os.path.exists(temp_fasta_path):
            os.remove(temp_fasta_path)

    print("\n" * 2)
    print("========================================")
    print("Processing Complete!")
    print(f"Successfully processed pairs: {success_count}")
    print(f"Failed pairs: {fail_count}")
    print(f"All outputs are saved in: {OUTPUT_DIR}")
    print("========================================")
if __name__ == '__main__':
    main()
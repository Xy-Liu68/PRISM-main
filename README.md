# PRISM-main
<!--# PRISM: A Structure-Guided Computational Approach for Identifying RNA-Targeting Small Molecules by Integrating 3D Conformational and Chemical Information-->

## Installation
### If you prefer a faster setup, you can use the provided environment.yaml file:
```bash

conda env create -f environment.yaml -y
conda activate prism

```

### You can install the environment either by following the step-by-step instructions below.
```bash

# Create a conda environment
conda create -y -n prism python=3.8
conda activate prism

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dgllife==0.3.2
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch_geometric==2.6.1
pip install egnn-pytorch==0.2.8
pip install rdkit==2022.3.5
pip install biopython==1.83
pip install biopandas==0.5.1
pip install forgi==2.2.3
pip install py3dmol==2.5.3
pip install pychimera==0.2.7
pip install transformers==4.28.1
pip install pandas==1.2.4
pip install scikit-learn==0.24.2
pip install scipy==1.10.1
pip install networkx==2.8.8
pip install joblib==1.4.2
pip install matplotlib==3.7.5 
pip install seaborn==0.13.2
pip install tensorboard==2.14.0 
pip install tensorboardx==2.6.2.2
```


## Data Preparation
### Generate RNA Structure
We use [RhoFold+](https://github.com/ml4bio/RhoFold) to generate RNA 3D Structure
Please save `RhoFold-main` to `./PRISM-main/Reference_module`


### Data Processing
### Training datasets convert to 3D
The training data is available in: `SMRTnet_data.txt` 
```bash
python process_smrtnet.py
```
files will be saved in the data folder: `./PRISM-main/3D_processed_output/smrtnet_processed_output`


### Benchmark datasets convert to 3D
The model generalization evaluation data is available in: `benchmark_all.txt`. It contains: `benchmark_NALDB.txt`, `benchmark_NewPub.txt`, `benchmark_RBIND.txt`, `benchmark_RSIM.txt`, `benchmark_SMMRNA.txt`.
```bash
python process_benchmark.py
```
files will be saved in the data folder: `./PRISM-main/3D_processed_output/all_benchmark_processed_output`


## Using PRISM
### Model Training
```bash
python train.py
```

### Evaluation of Model Generalization Ability
```bash
python train_evaluation.py
```

### Model Path
The weights of the model we have trained are all saved in: `./PRISM-main/results`


### Training your own datasets
We have simplified the process so you do NOT need secondary structure information (e.g., `((...))`).

There's two steps:
1. **Data Processing**: Generate 3D structures from sequences.
2. **Training**: Train the model using the processed data.

## 1. Data Preparation
Prepare your dataset in a `.txt` file using **Tab-separated values**.
The file must contain exactly **3 columns** in the following order:
1.  **SMILES**: The chemical structure of the small molecule.
2.  **RNA Sequence**: The nucleotide sequence (A, U, C, G).
3.  **Label**: Binary label (1 for binding, 0 for non-binding).
**Format Example:**
| SMILES | Sequence | label |

|-----------------|-------------|-------------|

| C#Cc1ccc(-c2nnc(NC(=O)c3ccc(N)cc3)o2)cc1 | UGGCACCUCGAUGUCGGCUCAUCACAUCCUG | 1 |
| C#CCCC1(CCNCc2c[nH]c3[nH]c(N)nc(=O)c23)N=N1 | CUGGGUCGCAGUAACCCCAGUUAACAAAACA | 0 | 
| CCOc1ccc(NC(=O)c2ccc(CNCc3ccc(O)cc3)cc2)cc1 | AAAGGUCGCAGUCCCCCCAGUUAACAAAAAA | 0 | 
...
| ... | ... | ... | ... | 

## 2. Step 1: Process Data (Generate 3D Structures)
Use process_new_data.py to convert your sequences into 3D PDB structures using RhoFold+ and process SMILES into graphs.
```bash
python ./PRISM-main/process_new_data.py \
    --input_file ./PRISM-main/data/{your_dataset}.txt \
    --output_dir ./PRISM-main/3D_processed_output/{your_dataset}_processed
```
--input_file: Path to your dataset text file.
--output_dir: Folder where the 3D structures will be saved.

Note: This step requires a GPU and may take time depending on dataset size.


## 3. Step 2: Train the Model
Once processing is complete, use train_new_data.py to train the model. 
```bash
python ./PRISM-main/train_new_data.py \
  --data_file ./PRISM-main/data/{your_dataset}.txt \
  --data_dir ./PRISM-main/3D_processed_output/{your_dataset}_processed \
  --epochs 100 \
  --batch_size 16 \
  --log_file ./PRISM-main/results/{your_dataset}_training_log.txt
```

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

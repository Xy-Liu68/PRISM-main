### PRISM Data Preparation
### Download Pre-processed 3D Data
The processed 3D structures are available for download from Google Drive.[prism_data](https://drive.google.com/drive/folders/1h8Zi4Dl5_dJ3-1Oz0pR1Elf06MDFtp2L?usp=drive_link)

### Unzip the datasets
```bash
unzip smrtnet_processed_output.zip -d ./PRISM-main/3D_processed_output

unzip all_benchmark_processed_output.zip -d ./PRISM-main/3D_processed_output
```

### Generate RNA Structure from Scratch
We use [RhoFold+](https://github.com/ml4bio/RhoFold) to generate RNA 3D Structure.It takes about 60 hours to run.

The training data is available in: `SMRTnet_data.txt` 
### Training datasets convert to 3D
```bash
python process_smrtnet.py
```
files will be saved in the data folder:`./PRISM-main/3D_processed_output/smrtnet_processed_output`


The model generalization evaluation data is available in: `benchmark_all.txt`. It contains: `benchmark_NALDB.txt`, `benchmark_NewPub.txt`, `benchmark_RBIND.txt`, `benchmark_RSIM.txt`, `benchmark_SMMRNA.txt`.
### Benchmark datasets convert to 3D
```bash
python process_benchmark.py
```
files will be saved in the data folder:`./PRISM-main/3D_processed_output/all_benchmark_processed_output`

1) R-BIND (https://rbind.chem.duke.edu/), `benchmark_RBIND.txt`
2) R-SIM (https://web.iitm.ac.in/bioinfo2/R_SIM/), `benchmark_RSIM.txt`
3) SMMRNA (http://www.smmrna.org/), `benchmark_SMMRNA.txt`
4) NALDB (http://bsbe.iiti.ac.in/bsbe/naldb/HOME.php), `benchmark_NALDB.txt`
5) NewPub (https://pubmed.ncbi.nlm.nih.gov/), `benchmark_NewPub.txt`

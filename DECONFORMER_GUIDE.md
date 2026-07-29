# Deconformer End-to-End Training Guide

This guide walks you through preparing data and training a Deconformer model from scratch. The workflow corresponds to the following files in this repository:

- `01_tsp_download/download_tsp_h5ad.sh`: Download single-cell reference data
- `02_tsp_preproc/tsp_preproc.ipynb`: Preprocess the single-cell reference `.h5ad`
- `04_run_deconvolution/05_deconformer/train/01_simulate.sh`: Generate simulated training data
- `04_run_deconvolution/05_deconformer/train/02_train.sh`: Train the Deconformer model

## Requirements

- **OS**: Linux/UNIX (recommended); Windows users may use WSL2
- **Python**: >= 3.10
- **Main dependencies**:
  - `torch >= 2.0.0`
  - `scanpy >= 1.9.3`
  - `h5py`
  - `numba`
  - `joblib`
  - `tqdm`
  - `pandas`
  - `numpy`

Installation example:

```bash
pip install torch scanpy h5py numba joblib tqdm pandas numpy
```

If you plan to train on GPU, make sure to install the CUDA-enabled version of PyTorch.

## Step 1: Download the single-cell reference data

Run the download script:

```bash
bash 01_tsp_download/download_tsp_h5ad.sh
```

This script downloads the raw Tabula Sapiens `.h5ad` file from the HCA Data Portal via its manifest API. After download, the file is typically located under `01_tsp_download/` or the current working directory, named `TabulaSapiens.h5ad`.

## Step 2: Preprocess the single-cell reference data

Open and run the preprocessing notebook:

```bash
02_tsp_preproc/tsp_preproc.ipynb
```

This notebook performs the following preprocessing steps:

- Reads the raw `TabulaSapiens.h5ad`
- Quality control and cell-type annotation cleanup
- Gene filtering and expression matrix normalization
- Outputs the preprocessed reference data as `02_tsp_preproc/TabulaSapiens_smartseq2.h5ad`

> Note: The preprocessing logic may vary slightly depending on the data version. Please follow the instructions embedded in the notebook.

## Step 3: Generate simulated training data

Navigate to the Deconformer training script directory:

```bash
cd 04_run_deconvolution/05_deconformer/train
```

Run the simulation script:

```bash
bash 01_simulate.sh
```

This script calls `Deconformer/deconformer_simulate.py`. Key parameters are shown below:

```bash
N_BATCH=160              # Number of simulation batches
N_SAMPLE_PER_BATCH=5000  # Samples per batch; total = 160 × 5000 = 800,000 samples
CHUNK_SIZE=200           # HDF5 chunk size
N_CPU=25                 # Number of parallel CPU cores

python ../Deconformer/deconformer_simulate.py \
    -i ../../../02_tsp_preproc/TabulaSapiens_10X.h5ad \
    --cell-types-file ../Deconformer/resource/NBT_simu_cell_order_sccpm.txt \
    --genes-file ../Deconformer/resource/tsp_mRNA_genes.txt \
    -n $N_BATCH -s $N_SAMPLE_PER_BATCH -c $CHUNK_SIZE -j $N_CPU
```

Output file:

- `simulated_data.h5`: HDF5 file containing the expression matrix, cell-fraction labels, gene names, cell-type names, and sample IDs.

### Key parameters of `deconformer_simulate.py`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i` / `--input-h5ad` | Path to input single-cell reference `.h5ad` | None |
| `-o` / `--output-dir` | Output directory | `./` |
| `--final-name` | Name of the output HDF5 file | `simulated_data.h5` |
| `--cell-types-file` | Cell-type order file | None |
| `--genes-file` | Target gene list file | None |
| `-n` | Number of simulation batches | `20` |
| `-s` | Samples per batch | `20000` |
| `-c` | Samples per HDF5 chunk | `1000` |
| `-j` | Number of CPU cores | All available |
| `--keep-temp` | Keep temporary chunk files | `False` |

## Step 4: Train the Deconformer model

In the `04_run_deconvolution/05_deconformer/train` directory, run:

```bash
bash 02_train.sh
```

This script calls `Deconformer/deconformer_train.py`. The core command is:

```bash
MODEL_DIR=demo_model
N_CPU=25

python ../Deconformer/deconformer_train.py \
    -i simulated_data.h5 \
    --gmt ../Deconformer/resource/c5.go.bp.v2023.1.Hs.symbols.gmt \
    --project-name $MODEL_DIR \
    --num-workers $N_CPU
```

### Key parameters of `deconformer_train.py`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i` / `--input-h5` | Path to input simulated HDF5 file | None |
| `--gmt` | Path to GOBP pathway GMT file | None |
| `--project-name` | Prefix for output directory | None |
| `--mask-file` | Pre-computed pathway-gene mask file | `None` (auto-computed) |
| `--loss` | Loss function: `MSE` or `MAE` | `MSE` |
| `--lr` | Learning rate | `0.0001` |
| `--batch-size` | Training batch size | `64` |
| `--epochs` | Number of training epochs | `20` |
| `--n-pathways` | Number of top pathways to use | `5000` |
| `--n-tvg` | Number of top variance genes to select | `10000` |
| `--dropout-rate` | Gene dropout rate for data augmentation | `0.2` |
| `--num-workers` | DataLoader workers for HDF5 reading | `10` |
| `--train-ratio` | Ratio of samples used for training | `0.9975` |

After training, the model files and intermediate results are saved in a directory named `demo_model_<timestamp>/`, which contains:

- `model_checkpoint_epoch_*.pt`: Model checkpoints
- `top_variance_genes_10000.txt`: List of top variance genes
- `mask_gene_5000_pathway*.txt.gz`: Pathway-gene mask
- `cell_types.tsv` / `mRNA_genes.tsv`: Cell-type and gene order files

## Step 5: Inference with the trained model (optional)

After training, you can use `Deconformer/deconformer_inference.py` to infer cell fractions for new samples:

```bash
cd 04_run_deconvolution/05_deconformer/Deconformer
python deconformer_inference.py \
    --model adult_model \
    --input example_input/PE2020.TPM.txt \
    --output inference_results/test_output.txt
```

To use a custom trained model, specify the model directory directly:

```bash
python deconformer_inference.py \
    --model-dir /path/to/demo_model_<timestamp> \
    --epoch 15 \
    --cell-types-file /path/to/cell_types.tsv \
    --genes-file /path/to/genes.txt \
    --input your_input.tsv \
    --output your_output.tsv \
    --device cuda
```

## Pre-trained Models in This Repository

The directory `04_run_deconvolution/05_deconformer/Deconformer/checkpoints/` already provides three pre-trained models:

- `adult_model`: 60 basic cell types
- `fetal_model`: 27 cell types + 3 trophoblast types + 4 fetal cell types
- `preg_model`: 60 cell types + early and late stages of SCT, EVT, VCT (6 trophoblast types in total)

These can be used directly for inference without retraining.

## Pipeline Overview

```
Download reference data
        ↓
Preprocess .h5ad
        ↓
Generate simulated data (deconformer_simulate.py)
        ↓
Train Deconformer (deconformer_train.py)
        ↓
Infer cell fractions (deconformer_inference.py)
```

## Notes

1. **Memory and GPU memory**: Simulation data generation mainly consumes CPU memory; during training, GPU memory usage at batch size 64 is below 30 GB, allowing training on consumer-grade GPUs such as the RTX 5090.
2. **HDF5 chunking**: Since the 2026-07-22 update, simulated data are output in HDF5 format. The chunking mechanism avoids loading the entire matrix into memory at once.
3. **Cell-type and gene order files**: The `cell-types-file` and `genes-file` must be consistent between simulation and training, and the same order files must be used during inference.
4. **Path adjustments**: Paths in this guide are relative to the repository root. Adjust relative paths according to your current working directory when running commands.

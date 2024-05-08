# Deconformer

preprint article : Pathway-enhanced Transformer-based model for robust enumeration of cell types from the cell-free transcriptome

[DOI: 10.1101/2024.02.28.582494](https://doi.org/10.1101/2024.02.28.582494)

## Deconformer model structure
![model structure](model_structure.png)

## Requirements

- **OS**: Linux/UNIX/Windows
- **Python Version**: >= 3.10.12
- **Library**:
  - torch >= 2.0.0
  - scanpy >= 1.9.3

## Usage

### Step 1: only use pre-trained the model to inference cfRNA samples

```python
python  deconformer_predict.py  saved_model_path  expression_profile 
```

#### Input:

- **expression_profile**: An expression profile of a cfRNA sample in CSV format, for which you need to infer the source scores. .
- **saved_model_path**: A path for saving pre-trained model parameters and mask matrices (for example, the adult model:  ./model_weight/adult_model/ ).

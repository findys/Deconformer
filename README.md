# Deconformer

preprint article : Pathway-enhanced Transformer-based model for robust enumeration of cell types from the cell-free transcriptome

[DOI: 10.1101/2024.02.28.582494](https://doi.org/10.1101/2024.02.28.582494)


the model weights used in article list in model_weights folder

model train:
for example config

ann = sc.read_h5ad("./resource/train_simu_sample_80w.h5ad")

gmt = "./resource/c5.go.bp.v2023.1.Hs.symbols.gmt"

project_name = "deconformer_train_save_files_"

run python deconformer_train

fitting started!


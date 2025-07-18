# Analysis Code in the Article

## Introduction

The analysis and visualizations presented in the article are based on the downstream analysis of cell composition fractions predicted by Deconformer on simulated or real cfRNA datasets. To ensure the authenticity and reproducibility of the analysis, Quarto documents—similar to Jupyter Notebooks—in RStudio were used.

## Files in the Directory

The directory contains three types of files: `*.qmd`, `*.sessionInfo`, and `.RData`.
- `*.qmd` files contain the analysis and visualization code.
- `*.sessionInfo` files record the computational environment used for the corresponding code.
- `.RData` files contain the data required for the analyses presented in the article.

Specifically:
- `test.*` corresponds to the analysis comparing deconvolution method performance.
- `test2.*` corresponds to the pathway perturbation analysis of Deconformer.
- `model_explain.*` corresponds to the analysis on real cfRNA datasets "Wang et al. (COVID-19)" and "Tao et al. (HCC)".
- `stat_preg_model.*` corresponds to the analysis on pregnancy-related real cfRNA datasets "Munchel et al. (PE)" and "SZMCHH cohort (GDM)".

## How to Run

Open the `.qmd` Quarto document in RStudio.

### Install Required R Packages

Different `.qmd` files require different R packages. First, open the `.qmd` file you want to run, and install all the necessary R packages mentioned in the first code chunk using CRAN, Bioconductor, or devtools.

### Load Packages and Import Data

Run the first code chunk to load the required R packages. Then, use `load()` to import the corresponding `.RData` data file.

### Run the Remaining Code Chunks Sequentially

Place the cursor in the first code chunk. Then click in the RStudio interface: `Run -> Run All Chunks Below`. The program will start executing from the second code chunk onward, running all subsequent chunks in order.
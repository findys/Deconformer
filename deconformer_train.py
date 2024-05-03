import pandas as pd
import scanpy as sc
from deconformer_model import fitting_pipelines
import sys

ann = sc.read_h5ad("./resource/train_simu_sample_80w.h5ad")
gmt = "./resource/c5.go.bp.v2023.1.Hs.symbols.gmt"
project_name = "deconformer_train_save_files_"

lr = 0.0005
batchsize = 128
loss = "MSE"

fitting_pipelines(project=project_name,data=ann, mask_file=None, gmtfile=gmt,loss= loss,learning_rate=lr,n_pathway=5000, batch_size=batchsize, epoch=20, n_tvg=10000)


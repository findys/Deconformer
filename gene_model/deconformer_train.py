import torch
import pandas as pd
import scanpy as sc
from deconformer_model import fitting_pipelines
import sys

# ann = sys.argv[1]
# gmt = sys.argv[2]
# project_name = sys.argv[3]

# ann = sc.read_h5ad("/home/share/huadjyin/home/wangchaoxing/yanshuo1/former_simulate/train_simu_samples_float32/train_simu_sample_60w.h5ad")
# gmt = "/home/share/huadjyin/home/wangchaoxing/deconformer/simulate/resource/c5.go.bp.v2023.1.Hs.symbols.gmt"
project_name = "gene_deconformer_new_train_save_files_"
data_path = ["/home/share/huadjyin/home/wangchaoxing/yanshuo1/former_simulate/train_simu_samples_float32/train_simu_sample_60w.h5ad",
            "/home/share/huadjyin/home/wangchaoxing/yanshuo1/former_simulate/train_simu_samples_float32/train_simu_sample_20w.h5ad"]
genelist_path = "/home/share/huadjyin/home/wangchaoxing/yanshuo1/former_simulate/train_simu_samples_float32/genelist.txt"


lr = 0.0005
batchsize = 128
loss = "MSE"


with open(genelist_path, "r") as f:
    genelist = [line.strip() for line in f]  # strip() 去除每行末尾的换行符
genelist = list(set(genelist))

fitting_pipelines(project=project_name,data_path=data_path,genelist=genelist,loss= loss,learning_rate=lr, batch_size=batchsize, epoch=60, n_tvg=10000)


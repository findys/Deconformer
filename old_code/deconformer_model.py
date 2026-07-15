



import time
import pandas as pd
import scanpy as sc
import gc
import numpy as np
import random
import os

from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import torch

import anndata
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler

from numba import jit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def top_variance_gene(adata,n_topgenes,outdir):
    variances = np.var(adata.X, axis=0)
    genes = adata.var_names
    df_var = pd.DataFrame({'variance': variances}, index=genes)
    df_sorted = df_var.sort_values(by='variance', ascending=False)
    df_vartop = df_sorted.head(n_topgenes)
    print("top variance genes ")
    print(df_vartop)
    df_vartop.to_csv(outdir + "top_variance_genes_" + str(df_vartop.shape[0]) + ".txt", sep='\t')
    return df_vartop


def read_pathway_gmtfile(file ,tvg ,th):
    with open(file, 'r') as f:
        genesets = {}
        genesets_rate = {}
        for line in f:
            split_line = line.strip().split('\t')
            geneset_name = split_line[0]
            genes = split_line[2:]
            rate = len(set(genes).intersection(set(tvg))) /(len(set(genes).union(set(tvg))))
            genesets_rate[geneset_name] = rate
            genesets[geneset_name] = genes
        rate_top = sorted(genesets_rate.items(), key=lambda item: item[1], reverse=True)[:th]
        result = [item[0] for item in rate_top]
        top_pathway = {key: genesets[key] for key in result}

        return genesets_rate, top_pathway


def pathway_mask(pathways, tvg, outdir):
    gene = list(pathways.values())
    flattened_list = [item for sublist in gene for item in sublist]
    uni_list = list(set(flattened_list))
    print("pathways genes count ")
    print(len(uni_list))

    inter_tvg = list(set(tvg).intersection(set(uni_list)))
    random.shuffle(inter_tvg)
    print("inter tvg pathway genes count ")
    print(len(inter_tvg))

    matrix = pd.DataFrame(np.zeros((len(inter_tvg), len(pathways))), index=inter_tvg,
                          columns=list(pathways.keys()))
    # 根据通路和基因的关系填充矩阵
    for pathway, gene_list in pathways.items():
        inter_gene = list(set(inter_tvg).intersection(set(gene_list)))
        matrix.loc[inter_gene, pathway] = int(1)
    matrix.to_csv(outdir + "mask_gene_" + str(matrix.shape[1]) + "_pathway" + str(matrix.shape[0]) + ".txt", sep='\t')

    return matrix



class deconformer(nn.Module):
    def __init__(self, num_cell_types, mask):
        super(deconformer, self).__init__()

        self.embedding_dim = 128
        self.num_heads = 4
        self.hidden_dim = 512
        self.num_layers = 4

        self.mask = mask
        self.embedding_matrix = nn.Parameter(torch.rand(self.mask.shape[1], self.embedding_dim))

        # 定义CLS token的embedding vector
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            num_encoder_layers=self.num_layers,
            activation='gelu',
            num_decoder_layers=0,
            dropout=0.1  # Dropout in Transformer
        )

        self.fraction = nn.Linear(self.embedding_dim, num_cell_types)

    def forward(self, x):
        x = x.repeat(1, self.mask.shape[0], 1)
        x = x * self.mask.to(x.device)
        x = x @ self.embedding_matrix

        # 将CLS token加入到输入矩阵的开始位置
        cls_token = self.cls_token_embedding.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        x = x.permute(1, 0, 2)
        x = self.transformer.encoder(x)
        cls_token = x[0]
        output = self.fraction(cls_token)
        softmax_output = torch.softmax(output, dim=1)
        return softmax_output



class Simu_Dataset(Dataset):
    def __init__(self, adata, input_genes):
        adata = adata[:, input_genes]
        self.expressions = adata.X
        self.labels = adata.obs.values

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        expression = torch.tensor(self.expressions[idx], dtype=torch.float32).unsqueeze(0).to(device)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).to(device)
        return expression, label


# 自定义损失函数
def mse_non_zero(y_pred, y_true):
    # 找出 y_true 中不为零的元素的位置
    non_zero_indices = torch.nonzero(y_true, as_tuple=True)
    # 根据位置取出对应的 y_true 和 y_pred
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    # 计算非零位置的 MSE
    mse_non_zero = torch.mean((y_true_non_zero - y_pred_non_zero) ** 2)
    return mse_non_zero


def mae_non_zero(y_pred, y_true):
    # 找出 y_true 中不为零的元素的位置
    non_zero_indices = torch.nonzero(y_true, as_tuple=True)
    # 根据位置取出对应的 y_true 和 y_pred
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    # 计算非零位置 MAE
    mae_non_zero = torch.mean(torch.abs(y_true_non_zero - y_pred_non_zero))
    return mae_non_zero




class WarmupConstantLRScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        # 在前 warmup_steps 步中线性增加学习率
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # 在 warmup_steps 之后保持学习率不变
            lr = self.max_lr

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1
        return lr


def ccc_per_sample(y_pred, y_true):
    ccc_values = torch.zeros(y_pred.size(0))  # 初始化存储每个样本CCC值的张量
    for i in range(y_pred.size(0)):
        x = y_pred[i].flatten()
        y = y_true[i].flatten()

        x_mean = x.mean()
        y_mean = y.mean()
        x_var = x.var(unbiased=False)
        y_var = y.var(unbiased=False)
        cov = (x - x_mean) * (y - y_mean)
        cov = cov.mean()
        ccc = (2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2)
        ccc_values[i] = ccc

    return ccc_values.mean()

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for expressions, frac in dataloader:
            outputs = model(expressions)
            if loss_fn == "MSE":
                loss = mse_non_zero(outputs, frac)
            else:  # 这里可以加入其他损失函数的条件
                loss = mse_non_zero(outputs, frac)  # 默认使用MSE作为备选
            total_loss += loss.item()
            count += 1
    average_loss = total_loss / count
    return average_loss



def train(model, loss_function, total_epoch, save_path, optimizer, dataloader, test_dataloader, scheduler):
    scaler = GradScaler()
    for epoch in range(total_epoch):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        pbar = tqdm(dataloader)
        for batch_idx, (expressions, frac) in enumerate(pbar):
            current_lr = scheduler.step()
            optimizer.zero_grad()
            with autocast():
                outputs = model(expressions)
                if loss_function == "MSE":
                    loss = mse_non_zero(outputs, frac)
                else:
                    loss = mae_non_zero(outputs, frac)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            batch_count += 1
            pbar.set_description(
                f"Epoch {epoch + 1} avgLoss: {loss_sum / batch_count:.5f},Loss: {loss.item():.5f}, LR: {current_lr:.5f}")

        # 在每个epoch结束时，评估测试数据集上的平均损失和CCC
        test_avg_loss= evaluate(model, test_dataloader, loss_function)
        print(f"Epoch {epoch + 1}, Test Avg Loss: {test_avg_loss:.5f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(save_path, f"model_checkpoint_epoch_{epoch + 1}.pt"))


def minmax_scale_per_cell(adata):
    min_values = np.min(adata.X, axis=1, keepdims=True)
    max_values = np.max(adata.X, axis=1, keepdims=True)

    adata.X = (adata.X - min_values) / (max_values - min_values)
    return adata

def df_2_ann(df_data):
    df_data_reindex = df_data.T
    adata = anndata.AnnData(X=df_data_reindex.values, var=pd.DataFrame(index=df_data_reindex.columns),
                            obs=pd.DataFrame(index=df_data_reindex.index))
    sc.pp.normalize_total(adata, target_sum=10000)
    return adata



def rdm_drop(adata, rate):
    n_rows, n_cols = adata.X.shape
    for i in range(n_rows):
        indices_to_zero = np.random.choice(n_cols, size=int(rate * n_cols), replace=False)
        adata.X[i, indices_to_zero] = 0


def fitting_pipelines(project, data, mask_file, gmtfile,loss="MSE",learning_rate=0.0005, n_pathway=5000, batch_size=128,
                      epoch=15,
                      n_tvg=10000):
    run_time = time.strftime("%m%d_%H%M%S")
    project = project or "project_"
    project = project + str(run_time)
    print("project name ")
    print(project)
    project_path = os.getcwd() + '/' + project + "/"
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)

    # tvg gene
     # q去掉全0无意义基因
    sc.pp.normalize_total(data, target_sum=10000)
    print(" simulate data tvg")
    non_zero_columns = np.sum(data.X, axis=0) != 0
    data = data[:, non_zero_columns]
    tvg = top_variance_gene(data, n_tvg, project_path).index.tolist()

    # top pathway
    all_pathway_rate, top_pathway = read_pathway_gmtfile(gmtfile, tvg, n_pathway)
    # pathway mask
    if mask_file is None:
        df_mask = pathway_mask(top_pathway, tvg, project_path)
    else:
        df_mask = pd.read_csv(mask_file, sep='\t', index_col=0)
    num_cell_types = data.obs.shape[1]
    mask = df_mask.T.to_numpy()
    input_genes = df_mask.index.tolist()
    mask_matrix = torch.from_numpy(mask).int()
    print("mask tensor ")
    print(mask_matrix)
    print("mask tensor shape : " + str(mask_matrix.shape))

    data_gene = data[:, input_genes]
    del data
    gc.collect()
    data_gene.X = np.log2(data_gene.X + 1)

    preps_train_data = data_gene
    del data_gene
    gc.collect()
    print("pre-processed data:")
    print(preps_train_data)

    rdm_drop(preps_train_data, 0.2)

    #split train set and test set
    n_obs = preps_train_data.n_obs
    # 计算两个部分的大小
    n_train = int(n_obs * 0.9975)
    n_test = n_obs - n_train
    # 生成一个随机的排列
    np.random.seed(0)  # 为了可重复性
    indices = np.random.permutation(n_obs)
    # 分割索引以创建两个部分
    indices_train = indices[:n_train]
    indices_test = indices[n_train:]
    train_ann = preps_train_data[indices_train]
    test_ann = preps_train_data[indices_test]

    print(train_ann)
    print(test_ann)

    # 创建 DataLoader 对象
    dataset = Simu_Dataset(train_ann, input_genes=input_genes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Simu_Dataset(test_ann, input_genes=input_genes)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    print("model build and initial")
    model = deconformer(num_cell_types, mask_matrix)

    # mutil-GPU parallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.to(device)
        model.module.embedding_matrix = model.module.embedding_matrix.to(device)
        model.module.mask = model.module.mask.to(device)
    else:
        model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    print('Number of parameters: ', num_parameters)
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', num_trainable_parameters)

    # 定义训练学习率调整策略
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #total_steps = len(dataloader)*epoch
    scheduler = WarmupConstantLRScheduler(optimizer, warmup_steps=10000, max_lr=learning_rate)
    t0 = time.time()
    train(model=model,loss_function=loss,total_epoch=epoch, save_path=project_path, optimizer=optimizer,dataloader=dataloader,test_dataloader = test_dataloader,scheduler=scheduler)
    print(time.time()-t0)


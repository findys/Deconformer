



import time
import pandas as pd
import scanpy as sc
import gc
import numpy as np
import random
import os
import pickle

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

from scgpt import tokenizer
import json
# from pympler import asizeof # optional: function print_mem


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def top_variance_gene(adata,n_topgenes,outdir):
def top_variance_gene(adata,n_topgenes,outdir,name):
    variances = np.var(adata.X, axis=0)
    genes = adata.var_names
    df_var = pd.DataFrame({'variance': variances}, index=genes)
    df_sorted = df_var.sort_values(by='variance', ascending=False)
    df_vartop = df_sorted.head(n_topgenes)
    print("top variance genes ")
    print(df_vartop)
    df_vartop.to_csv(outdir + "top_variance_genes_" + str(df_vartop.shape[0]) +
                     "." + name + ".txt", sep='\t')
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


# TODO: 三维 -> 二维
class deconformer(nn.Module):
    # def __init__(self, num_cell_types, mask):
    def __init__(self, num_cell_types, vocab):
        super(deconformer, self).__init__()

        self.embedding_dim = 128
        # self.embedding_dim_ = 512
        self.num_heads = 4
        self.hidden_dim = 512
        self.num_layers = 4

        self.vocab = vocab
        self.ntoken = len(self.vocab)
        self.n_bins = 51
        self.padding_idx = vocab["<pad>"]

        # self.mask = mask
        # self.to = tokens
        # self.embedding_matrix = nn.Parameter(torch.rand(self.mask.shape[1], self.embedding_dim))

        self.GeneEncoder_embedding = nn.Embedding(
            self.ntoken, self.embedding_dim, padding_idx=self.padding_idx
        )

        self.GeneEncoder_enc_norm = nn.LayerNorm(self.embedding_dim)


        self.ValueEncoder_embedding = nn.Embedding(
            self.n_bins, self.embedding_dim, padding_idx=self.padding_idx
        )

        self.ValueEncoder_enc_norm = nn.LayerNorm(self.embedding_dim)

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

    def forward(self, x, src):

        # print("forward i: ")
        # print(src.shape)
        # print(x.shape)

        src = src.long()
        src = self.GeneEncoder_embedding(src)  # (batch, seq_len, embsize)
        src = self.GeneEncoder_enc_norm(src)

        x = x.long()
        x = self.ValueEncoder_embedding(x)  # (batch, seq_len, embsize)
        x = self.ValueEncoder_enc_norm(x)

        # print("forward ii: ")
        # print(src.shape)
        # print(x.shape)

        # x = x.unsqueeze(2)
        x = src * x
        
        # x = x.repeat(1, self.mask.shape[0], 1)
        # x = x * self.mask.to(x.device)
        # x = x @ self.embedding_matrix
        # TODO: 模型结构

        # print("forward iii: ")
        # print(x.shape)

        # 将CLS token加入到输入矩阵的开始位置
        cls_token = self.cls_token_embedding.repeat(x.size(0), 1, 1)
        # print("forward iii: ")
        # print(cls_token.shape)
        x = torch.cat([cls_token, x], dim=1) # (batch, seq_len+1, embsize)

        # print(x.shape)
        x = x.permute(1, 0, 2) # (seq_len+1, batch, embsize)
        # print(x.shape)
        x = self.transformer.encoder(x)
        cls_token = x[0]
        output = self.fraction(cls_token)
        softmax_output = torch.softmax(output, dim=1)
        return softmax_output



class Simu_Dataset(Dataset):
    def __init__(self, adata, ids):
        # adata = adata[:, input_genes]
        self.expressions = adata.X
        self.labels = adata.obs.values
        self.ids = ids

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        # expression = torch.tensor(self.expressions[idx], dtype=torch.float32).unsqueeze(0).to(device)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).to(device)
        # ids = torch.tensor(self.ids[idx], dtype=torch.float32).unsqueeze(0).to(device)
        expression = torch.tensor(self.expressions[idx], dtype=torch.int32).to(device)
        ids = torch.tensor(self.ids[idx], dtype=torch.int32).to(device)
        return expression, label, ids
    
# class Simu_Dataset(Dataset):
#     def __init__(self, adata, input_genes):
#         adata = adata[:, input_genes]
#         self.expressions = adata.X
#         self.labels = adata.obs.values

#     def __len__(self):
#         return len(self.expressions)

#     def __getitem__(self, idx):
#         expression = torch.tensor(self.expressions[idx], dtype=torch.float32).unsqueeze(0).to(device)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32).to(device)
#         return expression, label


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
        for expressions, frac, ids in dataloader:
            outputs = model(expressions, ids)
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
        for batch_idx, (expressions, frac, ids) in enumerate(pbar):
            current_lr = scheduler.step()
            optimizer.zero_grad()
            with autocast():
                outputs = model(expressions, ids)
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

# TODO:
def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    来自scGPT/preprocess.py
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int32)
    return digits


# def print_mem(data, ids):
#     mem_used = (asizeof.asizeof(data) + asizeof.asizeof(ids)) / 1024 ** 3
#     print("Size of preprocessed data : " + str(mem_used) + " GB")



# def fitting_pipelines(project, data, mask_file, gmtfile,loss="MSE",learning_rate=0.0005, n_pathway=5000, batch_size=128,
def fitting_pipelines(project, data_path, genelist ,loss="MSE",learning_rate=0.0005,  batch_size=128,
                      epoch=15,
                      n_tvg=10000):
    run_time = time.strftime("%m%d_%H%M%S")
    project = project or "project_"
    project = project + str(run_time)
    print("project name: " + project)
    project_path = os.getcwd() + '/' + project + "/"
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)

    # generate a geneVocab
    print("Generating geneVocab ...")
    gene_vocab = tokenizer.GeneVocab(genelist, specials = ["<cls>", "<pad>"])
    # write vocab in json
    vocab_dict = {word: idx for idx, word in enumerate(gene_vocab.get_itos())}
    with open(f"{project_path}/gene_vocab.json", "w") as json_file:
        json.dump(vocab_dict, json_file)
    with open(f"{project_path}/genelist.txt", "w") as f:
        for item in genelist:
            f.write(str(item) + '\n')
    del vocab_dict, genelist
    gc.collect()


    path_pickle_list = []
    tvg_list = []

    for current_data_path in data_path:
        print("Preprocessing simudata: " + current_data_path)
        data = sc.read_h5ad(current_data_path)
        if not "num_cell_types" in locals():
            num_cell_types = data.obs.shape[1]
        elif num_cell_types != data.obs.shape[1]:
            print("ERROR: num_cell_types of simudata is not equal!!!")

        # tvg gene
        # q去掉全0无意义基因
        sc.pp.normalize_total(data, target_sum=10000)
        # print(" simulate data tvg")
        non_zero_columns = np.sum(data.X, axis=0) != 0
        data = data[:, non_zero_columns]
        tvg = top_variance_gene(data, n_tvg, project_path, current_data_path.replace("/", "+")).index.tolist()
        del non_zero_columns
        gc.collect()

        inter_tvg = list(set(tvg).intersection(set(data.var.index.tolist())))
        random.shuffle(inter_tvg)
        data_gene = data[:, inter_tvg]
        tvg_list = list(set(tvg_list) | set(inter_tvg))
        del data, inter_tvg
        gc.collect()
        data_gene.X = np.log2(data_gene.X + 1)


        # binning
        n_bins = 51 # 分箱数
        print("Binning gene expression data ...")

        binned_rows = []
        if_log_zeros = False # 首次提示
        # import pdb;pdb.set_trace()

        for row in data_gene.X:
            if row.max() <= 0 and not if_log_zeros:
                print(
                    "WARNINGS: The input data contains rows of zeros. Please make sure this is expected."
                )
                binned_rows.append(np.zeros_like(row, dtype=np.int32))
                if_log_zeros = True
                continue

            # 表达非全0情况下
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            non_zero_digits = _digitize(non_zero_row, bins) # scgpt中特殊的digitize处理
            binned_row = np.zeros_like(row, dtype=np.int32)
            binned_row = np.array(binned_row)
            binned_row[non_zero_ids] = non_zero_digits
            binned_rows.append(binned_row)
        # end for row in data_gene.X

        data_gene.X = np.stack(binned_rows)
        del binned_rows, bins, non_zero_ids, non_zero_row, row
        gc.collect()


        # Tokenize and pad batch
        print("Tokenize and pad batch ...")
        tokens_list = [gene_vocab[token] for token in data_gene.var_names.tolist()]
        tokenize_res = tokenizer.tokenize_and_pad_batch(data = np.array(data_gene.X), 
                                                gene_ids = np.array(tokens_list), 
                                                max_len = 3000, 
                                                vocab = gene_vocab, 
                                                pad_token = "<pad>", 
                                                pad_value = gene_vocab["<pad>"], 
                                                append_cls = False)
        
        preps_train_data = sc.AnnData(X = tokenize_res["values"].numpy(), obs = data_gene.obs)
        preps_train_ids = tokenize_res["genes"].numpy() # np.array

        del data_gene, tokenize_res, tokens_list
        gc.collect()

        # optional: print var size
        # print_mem(preps_train_data, preps_train_ids)
        # mem_used = (asizeof.asizeof(preps_train_data) + asizeof.asizeof(preps_train_ids)) / 1024 ** 3
        # print("Size of preprocessed data : " + str(mem_used) + " GB")


        # pickle save (preps_train_data, preps_train_ids)
        name_pickle = "preprocessed_data." + current_data_path.replace("/", "+") + ".pkl"
        path_pickle = project_path + name_pickle
        print("Saving preprocessed data: " + name_pickle)
        with open(path_pickle, 'wb') as f:
            pickle.dump((preps_train_data, preps_train_ids), f)
        path_pickle_list.append(path_pickle)
        del preps_train_data, preps_train_ids
        gc.collect()

    # end for current_data_path in data_path
    with open(f"{project_path}/genelist_tvg.txt", "w") as f:
        for item in tvg_list:
            f.write(str(item) + '\n')


    # merge all data
    print("Merging preprocessed data ...")
    print("Loading preprocessed data: " + path_pickle_list[0])
    with open(path_pickle_list[0], 'rb') as f:
        preps_train_data, preps_train_ids = pickle.load(f)  # pickle load (preps_train_data, preps_train_ids)
    count_load = 0
    preps_train_data.obs.index = [str(count_load) + "_" + str(idx) for idx in preps_train_data.obs.index]
    count_load = count_load + 1
    # print_mem(preps_train_data, preps_train_ids)
    # mem_used = (asizeof.asizeof(preps_train_data) + asizeof.asizeof(preps_train_ids)) / 1024 ** 3
    # print("Size of preprocessed data : " + str(mem_used) + " GB")

    if len(path_pickle_list) > 1:
        for current_path_pickle in path_pickle_list[1:]:
            print("Loading preprocessed data: " + current_path_pickle)
            with open(current_path_pickle, 'rb') as f:
                preps_train_data_new, preps_train_ids_new = pickle.load(f)  # pickle load (preps_train_data, preps_train_ids)
            preps_train_data_new.obs.index = [str(count_load) + "_" + str(idx) for idx in preps_train_data_new.obs.index]
            count_load = count_load + 1

            # concatenate
            preps_train_data = anndata.concat([preps_train_data, preps_train_data_new])
            preps_train_ids = np.concatenate((preps_train_ids, preps_train_ids_new), axis=0)

            del preps_train_data_new, preps_train_ids_new
            gc.collect()
            # print_mem(preps_train_data, preps_train_ids)
            # mem_used = (asizeof.asizeof(preps_train_data) + asizeof.asizeof(preps_train_ids)) / 1024 ** 3
            # print("Size of preprocessed data : " + str(mem_used) + " GB")

    # end for current_path_pickle in path_pickle_list
    # import pdb;pdb.set_trace()


    print("pre-processed data:")
    print(preps_train_data)
    print(preps_train_data.X.shape)
    print(preps_train_ids)
    print(preps_train_ids.shape)
    print("pre-processed data END!")

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
    train_ids = preps_train_ids[indices_train]
    test_ids = preps_train_ids[indices_test]

    del preps_train_ids, preps_train_data
    gc.collect()
    

    print(train_ann)
    print(test_ann)

    # 创建 DataLoader 对象
    # dataset = Simu_Dataset(train_ann, input_genes=input_genes)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # test_dataset = Simu_Dataset(test_ann, input_genes=input_genes)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    dataset = Simu_Dataset(train_ann, train_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Simu_Dataset(test_ann, test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    print("model build and initial")
    # TODO:
    # model = deconformer(num_cell_types, mask_matrix)
    model = deconformer(num_cell_types, gene_vocab)

    # mutil-GPU parallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.to(device)
        # model.module.embedding_matrix = model.module.embedding_matrix.to(device)
        # model.module.mask = model.module.mask.to(device)
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


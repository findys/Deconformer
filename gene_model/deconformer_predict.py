import torch

import os
import sys
import re
import time

import pandas as pd
import anndata
import numpy as np
from deconformer_model import deconformer, _digitize
import scanpy as sc
from scgpt import tokenizer

save_file_dir = sys.argv[1]
pred_data = sys.argv[2]
load_check_point = sys.argv[3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
# def load_model(model_path, device, num_cell_types, mask_matrix, vocab):
def load_model(model_path, device, num_cell_types, vocab):
    model = deconformer(num_cell_types, vocab).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    #model.load_state_dict(checkpoint['model_state_dict'])
    return model

# 预测函数
def predict(model, data, ids, device,num_cell_types):
    model.eval()
    with torch.no_grad():
        pre_frac = np.zeros((data.shape[0], num_cell_types))
        row_index = 0
        for i in range(data.X.shape[0]):
            print(f"Sample Name: {data.obs_names[i]}")
            exp = data.X[i, :]
            ids_ = ids[i, :]
            t0 = time.time() 
            expressions = torch.tensor(exp, dtype=torch.int32).unsqueeze(0).to(device)
            ids_tokens = torch.tensor(ids_, dtype=torch.int32).unsqueeze(0).to(device)
            predictions = model(expressions, ids_tokens)
            if torch.cuda.is_available():
                pre_frac[row_index, :] = predictions.cpu().numpy()
            else:
                pre_frac[row_index, :] = predictions.numpy()
            print(time.time()-t0)
            row_index+=1
    return pre_frac

def minmax_scale_per_cell(adata):
    min_values = np.min(adata.X, axis=1, keepdims=True)
    max_values = np.max(adata.X, axis=1, keepdims=True)
    adata.X = (adata.X - min_values) / (max_values - min_values)
    return adata

def norm_real_data(df_data,gene):
    column_sums = df_data.sum()

    # 标准化每列的值，使得每列的总和为 10000
    df_data = df_data * 10000 / column_sums   

    hvg =gene
    new_index = pd.Index(hvg)
    df_reindexed = df_data.reindex(new_index, fill_value=0)

    df_data_reindex = df_reindexed.T

    adata = anndata.AnnData(X=df_data_reindex.values, var=pd.DataFrame(index=df_data_reindex.columns),
                            obs=pd.DataFrame(index=df_data_reindex.index))
    return adata


# def loadmodel_pred(file_dir,trained_model,mask,pre_data,cell_type,device, gene_vocab):
def loadmodel_pred(file_dir,trained_model,pre_data,cell_type,device, gene_vocab):
    # 加载模型
    print("load model : ")
    saved_model = file_dir + trained_model  # 你模型的路径
    num_cell_types = len(cell_type)  # cell类型数量
    # df_mask = mask
    # mask_matrix = torch.from_numpy(df_mask.T.to_numpy()).int().to(device)
    # loaded_model = load_model(saved_model, device, num_cell_types, mask_matrix, gene_vocab)
    loaded_model = load_model(saved_model, device, num_cell_types, gene_vocab)
    # 加载数据
    adata = pre_data  # 你的数据路径
    # adata = adata[:, df_mask.index.tolist()]
    adata.X = np.log2(adata.X + 1)
    # scaled_adata = adata
    # import pdb;pdb.set_trace()

    n_bins = 51 # 分箱数
    binned_rows = []
    if_log_zeros = False # 首次提示
    for row in adata.X:
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
    # end for

    adata.X = np.stack(binned_rows)
    tokens_list = [gene_vocab[token] for token in adata.var_names.tolist()]
    tokenize_res = tokenizer.tokenize_and_pad_batch(data = np.array(adata.X), 
                                              gene_ids = np.array(tokens_list), 
                                              max_len = 3000, 
                                              vocab = gene_vocab, 
                                              pad_token = "<pad>", 
                                              pad_value = gene_vocab["<pad>"], 
                                              append_cls = False)
    preps_data = sc.AnnData(X = tokenize_res["values"].numpy(), obs = adata.obs)
    preps_ids = tokenize_res["genes"].numpy()
    # 使用加载的模型进行预测
    pre_fraction = predict(loaded_model, preps_data, preps_ids, device,num_cell_types)
    df_pre = pd.DataFrame(pre_fraction, index=adata.obs_names.tolist(), columns=cell_type)
    print(df_pre)

    return df_pre


def get_mask_model(file_path,check_piont):
    # global maskm
    global model_pt
    for i in os.listdir(file_path):
        # if "mask" in i:
        #     maskm = i
        if "checkpoint" in i and str(check_piont) in i:
            model_pt =i
    # return model_pt,maskm
    return model_pt


with open(save_file_dir + "/genelist_tvg.txt", "r") as f:
    genes = [line.strip() for line in f]  # strip() 去除每行末尾的换行符
genes = list(set(genes))
print(genes)
# df_genes = pd.read_csv("./resource/tsp_mRNA_genes.txt",sep='\t',index_col=0)
# print(df_genes)
# genes =df_genes.index.tolist()

cell_type =  pd.read_csv("./resource/NBT_simu_cell_order_sccpm.txt",sep='\t',index_col=0).index.tolist()
# model_pt,mask = get_mask_model(save_file_dir, load_check_point)
model_pt = get_mask_model(save_file_dir, load_check_point)
print(model_pt)
# print(mask)
df_pred = pd.read_csv(pred_data,sep='\t',index_col=0)
# df_mask = pd.read_csv(save_file_dir + mask, sep='\t', index_col=0)
# import pdb;pdb.set_trace()
ann_pred = norm_real_data(df_pred,genes)
gene_vocab = tokenizer.GeneVocab.from_file(save_file_dir + "/gene_vocab.json")
# import pdb;pdb.set_trace()


df_pre= loadmodel_pred(save_file_dir,model_pt,ann_pred,cell_type,device, gene_vocab)
if "/" in pred_data:
    df_pre.to_csv("./inference_results/"+pred_data.split("/")[-1].split(".")[0]+"_deconformer_adult_re.txt", sep='\t')
else:
    df_pre.to_csv("./inference_results/" + pred_data.split(".")[0] + "_deconformer_adult_re.txt",sep='\t')

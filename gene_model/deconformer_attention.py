





import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import anndata
import scanpy as sc
import sys

save_pt = sys.argv[1]
pred_data = sys.argv[2]
out_dir = sys.argv[3]
CLS_att = sys.argv[4]

torch.set_printoptions(precision=10)


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


class Deconformer(nn.Module):
    def __init__(self, num_cell_types, mask):
        super(Deconformer, self).__init__()
        self.embedding_dim = 128
        self.num_heads = 4
        self.hidden_dim = 512
        self.num_layers = 4
        self.mask = mask
        self.embedding_matrix = nn.Parameter(torch.rand(self.mask.shape[1], self.embedding_dim))
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            num_encoder_layers=self.num_layers,
            dropout=0.1,
            num_decoder_layers=0,
            activation='gelu',
            norm_first=False  # 确保 norm_first 参数为 False
        )

        self.fraction = nn.Linear(self.embedding_dim, num_cell_types)

    def forward(self, x):
        x = x.repeat(1, self.mask.shape[0], 1)
        x = x * self.mask.to(x.device)
        x = x @ self.embedding_matrix
        cls_token = self.cls_token_embedding.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x.permute(1, 0, 2)

        attention_scores = []
        for layer in self.transformer.encoder.layers:
            # 应用自注意力
            attn_output, attn_weight = layer.self_attn(x, x, x, need_weights=True)
            attention_scores.append(attn_weight)
            # 应用第一个残差连接和归一化
            x = layer.norm1(x + attn_output)
            # 应用前馈网络
            x_ffn = layer.linear1(x)
            x_ffn = layer.dropout(F.gelu(x_ffn))
            x_ffn = layer.linear2(x_ffn)
            # 应用第二个残差连接和归一化
            x = layer.norm2(x + x_ffn)

        # Final processing
        cls_token = x[0]
        output = self.fraction(cls_token)
        softmax_output = torch.softmax(output, dim=1)
        return softmax_output,attention_scores[-1][0]


def get_attention(model, data, device,cell_types,CLS,out_dir):
    model.eval()
    with torch.no_grad():
        if CLS ==False:
            pre_frac = np.zeros((data.shape[0], len(cell_types)))
            row_index = 0
            for name, row_data in zip(data.obs_names, data.X):
                print(f"sample: {row_index}")
                print(f"Name: {name}")
                print(f"Data: {row_data}")
                expressions = torch.tensor(row_data, dtype=torch.float32).to(device)
                predictions, attention_score = model(expressions)
                pre_frac[row_index, :] = predictions.cpu().numpy()
                print(attention_score.shape)
                df_att = pd.DataFrame(attention_score.cpu().numpy(), index=[["cls_token"] + df_mask.columns.tolist()],
                                      columns=[["cls_token"] + df_mask.columns.tolist()])
                df_att.to_csv(out_dir + name + "_attention_matrix.txt", sep='\t')
                row_index += 1
        else:
            pre_frac = np.zeros((data.shape[0], len(cell_types)))
            cls_attention_df = pd.DataFrame()  # 初始化用于保存cls token attention的DataFrame
            row_index = 0
            for name, row_data in zip(data.obs_names, data.X):
                print(f"Name: {name}")
                print(f"Data: {row_data}")
                expressions = torch.tensor(row_data, dtype=torch.float32).to(device)
                predictions, attention_score = model(expressions)
                cls_attention = attention_score[0].cpu().numpy()
                cls_attention_df[name] = cls_attention  # 将当前样本的cls token attention添加到DataFrame
                row_index += 1

            # 保存所有样本的cls token attention信息
            cls_attention_df = cls_attention_df.T
            cls_attention_df.columns = [["cls_token"] + df_mask.columns.tolist()]
            cls_attention_df.to_csv(out_dir + name + "_attention_CLS.txt", sep='\t')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not out_dir.endswith('/'):
    out_dir += '/'

if str.lower(CLS_att) =="true":
    CLS_att = True
else:
    CLS_att= False

print(bool(CLS_att))

    
df_genes = pd.read_csv("./resource/tsp_mRNA_genes.txt",sep='\t',index_col=0)
print(df_genes)
genes =df_genes.index.tolist()

cell_type =  pd.read_csv("./resource/NBT_simu_cell_order_sccpm.txt",sep='\t',index_col=0).index.tolist()

# 加载mask
mask_path = save_pt+'/mask_gene_5000_pathway8886.txt'
df_mask = pd.read_csv(mask_path, sep='\t', index_col=0)
mask_matrix = torch.from_numpy(df_mask.T.to_numpy()).int().to(device)

# 初始化模型
model = Deconformer(num_cell_types=60, mask=mask_matrix).to(device)
# 加载模型状态
checkpoint_path = save_pt+"/model_checkpoint_epoch_15.pt"
checkpoint = torch.load(checkpoint_path)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict)
print("load done")


df_pred = pd.read_csv(pred_data,sep='\t',index_col=0)
ann_pred = norm_real_data(df_pred,genes)

adata = ann_pred[:, df_mask.index.tolist()]

adata.X = np.log2(adata.X + 1)
# scaled_adata = minmax_scale_per_cell(adata)
scaled_adata = adata
# 使用加载的模型进行预测
get_attention(model, scaled_adata, device,cell_type,CLS_att,out_dir=out_dir)



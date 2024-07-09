



import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from numba import jit
import time
from joblib import Parallel, delayed
import sys

# 动态加载指定目录下的.h5ad文件
def load_data_from_directory(directory_path,gene_order):
    cell_data_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".h5ad"):
            cell_type = filename.split('.')[0]  # 假设文件名格式为“细胞类型.h5ad”
            file_path = os.path.join(directory_path, filename)
            ann = sc.read_h5ad(file_path)
            ann = ann[:, genes]
            print(ann)
            if isinstance(ann.X,np.ndarray):
                cell_np = ann.X
            else:
                cell_np = ann.X.toarray()
            cell_data_dict[cell_type] = cell_np
    return cell_data_dict

# 使用Numba加速的随机生成细胞类型比例函数
@jit(nopython=True)
def generate_frac(sample_composition=60):
    random_vector = np.random.random(size=(sample_composition,))
    vector_sum = np.sum(random_vector)
    normalized_vector = random_vector / vector_sum
    frac = np.zeros(60)
    frac[:sample_composition] = normalized_vector
    np.random.shuffle(frac)
    return frac



def each_cell_contribution(cell_data, frac):
    # 确定样本数量，限制在200到800之间，以及不超过cell_data中的细胞数量
    n_cells = cell_data.shape[0]
    # 随机确定样本数量，范围在200到800之间
    sample_counts = np.random.randint(200, 800)
    # 如果随机确定的样本数量超过实际细胞数量，使用实际细胞数量作为样本数量
    if sample_counts > n_cells:
        sample_counts = n_cells
    # 随机选择细胞样本的索引
    random_cells_idx = np.random.choice(n_cells, sample_counts, replace=False)
    # 计算所选细胞的基因表达量的平均值
    exp = cell_data[random_cells_idx, :].mean(axis=0)
    # 根据细胞比例调整表达量
    contribute = exp * frac
    return contribute


# 模拟cfRNA样本
def simulate_samples(cell_data_dict,cell_type_order,genes, batch,c,n_samples=2000):
    all_sample = np.zeros((n_samples, len(genes)))
    df_y_s = []
    tags = []
    t0 = time.time()
    for sample_index in range(n_samples):
        tag = "sample_"+str(batch)+"_"+str(c)+"_"+str(sample_index)
        if sample_index % 500 ==0:
            print(tag)
            print(time.time()-t0)
        tags.append(tag)
        frac = generate_frac(len(cell_data_dict))
        df_frac = pd.DataFrame(frac.transpose(), index=cell_type_order, columns=[tag])
        df_y_s.append(df_frac)
        #sample_exp = np.zeros((1, len(genes)))

        cell_frac = dict(zip(cell_type_order, frac))
        sample_exp = np.zeros((1, len(genes)))
        for i in cell_frac.items():
            if i[1] == 0:
                continue
            contribute = each_cell_contribution(cell_data_dict[i[0]], i[1])
            sample_exp += contribute
        all_sample[sample_index, :] = sample_exp
        #print(time.time() - t0)
    df_y = pd.concat(df_y_s,axis=1)
    df_var = pd.DataFrame(genes, index=genes, columns=["genes"])
    df_obs = pd.DataFrame(tags, index=tags, columns=["sample_tag"])
    all_ann = anndata.AnnData(X=all_sample, var=df_var, obs=df_obs)
    return df_y,all_ann


def simulate(cell_data_dict,cell_type_order,genes,simu_batch,n_sample,save_path):
    df_merge_y = pd.DataFrame()
    ann_merge = anndata.AnnData()
    once_simu = int(n_sample/10)
    for i in range(10):
        sub_y, sub_ann = simulate_samples(cell_data_dict,cell_type_order,genes,simu_batch,i,n_samples=once_simu)
        df_merge_y = pd.concat([df_merge_y, sub_y], axis=1)
        var = sub_ann.var
        if ann_merge.n_obs == 0:
            ann_merge = sub_ann
        else:
            ann_merge = anndata.concat([ann_merge, sub_ann])
        ann_merge.var = var

    gene_exp = ann_merge.X
    ann_simi = anndata.AnnData(X=gene_exp, var=ann_merge.var, obs=df_merge_y.T)
    print(ann_simi)
    print(ann_simi.var)
    print(ann_simi.obs)
    save_time = time.strftime("%m%d%H%M%S")
    ann_simi.write_h5ad(save_path+"simulated_2w_"+str(simu_batch)+"_"+str(save_time)+".h5ad")

# 设置参数
cell_data_directory =  "./resource/single_cell_reference_data/" #
simulate_save_path = "./resource/simulated_datas/"

df_cell_types = pd.read_csv("./resource/NBT_simu_cell_order_sccpm.txt",sep='\t',index_col=0)
cell_types_order = df_cell_types.index.tolist()
print(df_cell_types)
genes = pd.read_csv("./resource/tsp_mRNA_genes.txt", sep='\t', index_col=0).index.tolist()
print(genes[:10])
cell_data = load_data_from_directory(cell_data_directory,genes)
print(cell_data.keys())

# 定义并行任务的数量
n_simulations = 20  # 比如说我们想并行运行20个模拟
n_samples_per_simulation = 20000  # 每个模拟生成的样本数

tp = time.time()
# 使用joblib并行执行
Parallel(n_jobs=n_simulations)(  # n_jobs=-1 表示使用所有可用的CPU核心
    delayed(simulate)(cell_data,cell_types_order,genes,simu_batch,n_samples_per_simulation,simulate_save_path)
    for simu_batch in range(n_simulations)
)

print(time.time()-tp)

import torch
import os
import sys
import time
import argparse
import pandas as pd
import anndata
import numpy as np

# 假设 deconformer_model.py 在同一目录下
from deconformer_model import deconformer
import scanpy as sc

# ================= 配置区域 (对应原 bash 中的 model_list.txt) =================
# 格式：'model_name': { 'dir': 模型目录，'epoch': checkpoint 标识，'cell_types': 细胞类型文件，'genes': 基因文件 }
# 注意：路径是相对于脚本所在目录还是绝对路径？这里假设传入的模型名对应相对路径结构，
# 或者你可以在运行时根据脚本位置动态调整。
# 为了兼容原 bash 逻辑，我们定义基础配置，实际路径会在主函数中结合脚本所在目录拼接。

MODEL_CONFIGS = {
    "adult_model": {
        "sub_dir": "model_weights/adult_model/",
        "epoch": "15",
        "cell_types_file": "NBT_simu_cell_order_sccpm.txt",
        "genes_file": "tsp_mRNA_genes.txt"
    },
    "fetal_model": {
        "sub_dir": "model_weights/fetal_model/",
        "epoch": "15",
        "cell_types_file": "fetal_simu_cell_order_1204.txt",
        "genes_file": "tsp_mRNA_genes.txt"
    },
    "preg_model": {
        "sub_dir": "model_weights/preg_model/",
        "epoch": "9",
        "cell_types_file": "cell_types.tsv",
        "genes_file": "mRNA_genes.tsv"
    }
}

# ================= 核心功能函数 (源自你提供的代码) =================

def load_model(model_path, device, num_cell_types, mask_matrix):
    model = deconformer(num_cell_types, mask_matrix).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 处理可能的 module. 前缀
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

def predict(model, data, device, num_cell_types):
    model.eval()
    pre_frac = np.zeros((data.shape[0], num_cell_types))
    row_index = 0
    
    with torch.no_grad():
        for i in range(data.X.shape[0]):
            print(f"Sample Name: {data.obs_names[i]}")
            exp = data.X[i, :]
            t0 = time.time()
            
            expressions = torch.tensor(exp, dtype=torch.float32).to(device)
            predictions = model(expressions)
            
            if torch.cuda.is_available():
                pre_frac[row_index, :] = predictions.cpu().numpy()
            else:
                pre_frac[row_index, :] = predictions.numpy()
            
            # 可选：打印耗时，生产环境可注释掉以免日志过多
            # print(f"Inference time: {time.time()-t0:.4f}s") 
            row_index += 1
    return pre_frac

def norm_real_data(df_data, gene):
    column_sums = df_data.sum()
    # 避免除以零
    column_sums[column_sums == 0] = 1 
    
    # 标准化每列的值，使得每列的总和为 10000
    df_data = df_data * 10000 / column_sums   

    new_index = pd.Index(gene)
    df_reindexed = df_data.reindex(new_index, fill_value=0)
    df_data_reindex = df_reindexed.T

    adata = anndata.AnnData(
        X=df_data_reindex.values, 
        var=pd.DataFrame(index=df_data_reindex.columns),
        obs=pd.DataFrame(index=df_data_reindex.index)
    )
    return adata

def get_mask_model(file_path, check_point):
    maskm = None
    model_pt = None
    for i in os.listdir(file_path):
        if "mask" in i:
            maskm = i
        if "checkpoint" in i and str(check_point) in i:
            model_pt = i
    
    if not maskm or not model_pt:
        raise FileNotFoundError(f"Could not find mask or checkpoint file containing '{check_point}' in {file_path}")
        
    return model_pt, maskm

def run_inference(model_dir, epoch_str, cell_types_file, genes_file, exp_tsv, out_tsv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载基因和细胞类型列表
    df_genes = pd.read_csv(genes_file, sep='\t', index_col=0)
    genes = df_genes.index.tolist()
    
    df_cell_types = pd.read_csv(cell_types_file, sep='\t', index_col=0)
    cell_type = df_cell_types.index.tolist()
    num_cell_types = len(cell_type)
    
    # 2. 查找具体的模型文件和 Mask 文件
    model_pt_name, mask_name = get_mask_model(model_dir, epoch_str)
    print(f"Using model checkpoint: {model_pt_name}")
    print(f"Using mask file: {mask_name}")
    
    saved_model_path = os.path.join(model_dir, model_pt_name)
    mask_file_path = os.path.join(model_dir, mask_name)
    
    # 3. 加载 Mask
    df_mask = pd.read_csv(mask_file_path, sep='\t', index_col=0)
    mask_matrix = torch.from_numpy(df_mask.T.to_numpy()).int().to(device)
    
    # 4. 加载并预处理输入数据
    df_pred = pd.read_csv(exp_tsv, sep='\t', index_col=0)
    ann_pred = norm_real_data(df_pred, genes)
    
    # 确保数据列与 Mask 索引一致
    ann_pred = ann_pred[:, df_mask.index.tolist()]
    ann_pred.X = np.log2(ann_pred.X + 1)
    
    # 5. 加载模型
    print("Loading model...")
    loaded_model = load_model(saved_model_path, device, num_cell_types, mask_matrix)
    
    # 6. 执行预测
    print("Starting inference...")
    pre_fraction = predict(loaded_model, ann_pred, device, num_cell_types)
    
    # 7. 保存结果
    df_pre = pd.DataFrame(pre_fraction, index=ann_pred.obs_names.tolist(), columns=cell_type)
    
    # 确保输出目录存在
    out_dir = os.path.dirname(out_tsv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    df_pre.to_csv(out_tsv, sep='\t')
    print(f"Inference complete. Results saved to: {out_tsv}")

# ================= 主入口 (替代 bash 逻辑) =================

def main():
    parser = argparse.ArgumentParser(
        description="Deconformer Prediction Tool (Python Native)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Available models:
  adult_model   : 60 basic cell types
  fetal_model   : 27 types of cells + 3 types of trophoblast cells + 4 types of fetal cells
  preg_model    : 60 types of cells + early and late stages of SCT, EVT, VCT, totaling six types of trophoblasts

Example usage:
  python deconformer_predict.py --model adult_model --input example_input/PE2020.TPM.txt --output inference_results/test_output.txt
        """
    )
    
    parser.add_argument('--model', '-m', type=str, required=True, 
                        help='Name of the trained model (e.g., adult_model, fetal_model, preg_model)')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to the input expression matrix TSV file')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Path to save the output inference result TSV file')
    
    args = parser.parse_args()
    
    # 验证模型名称
    if args.model not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        print(f"ERROR: '{args.model}' is not a valid model name.")
        print(f"Available models: {available}")
        sys.exit(1)
    
    # 获取脚本所在目录，用于构建相对路径 (模拟原 bash 的 $(dirname $0))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config = MODEL_CONFIGS[args.model]
    
    # 构建完整路径
    model_base_dir = os.path.join(script_dir, config['sub_dir'])
    cell_types_path = os.path.join(model_base_dir, config['cell_types_file'])
    genes_path = os.path.join(model_base_dir, config['genes_file'])
    
    # 检查必要文件是否存在
    if not os.path.exists(model_base_dir):
        print(f"ERROR: Model directory not found: {model_base_dir}")
        sys.exit(1)
    if not os.path.exists(cell_types_path):
        print(f"ERROR: Cell types file not found: {cell_types_path}")
        sys.exit(1)
    if not os.path.exists(genes_path):
        print(f"ERROR: Genes file not found: {genes_path}")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # 设置环境变量 (对应原 bash 的 export OMP_NUM_THREADS=20)
    os.environ["OMP_NUM_THREADS"] = "20"
    
    print(f"=== RUN: {args.model} {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    try:
        run_inference(
            model_dir=model_base_dir,
            epoch_str=config['epoch'],
            cell_types_file=cell_types_path,
            genes_file=genes_path,
            exp_tsv=args.input,
            out_tsv=args.output
        )
        print(f"=== DONE!!! {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    except Exception as e:
        print(f"=== ERROR: Inference failed. {str(e)} ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
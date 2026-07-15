import torch
import os
import sys
import time
import argparse
import pandas as pd
import anndata
import numpy as np
from tqdm import tqdm

# 假设 deconformer_model.py 在同一目录下
from deconformer_model import deconformer
import scanpy as sc

# ================= 配置区域 =================
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

# ================= 核心功能函数 =================

def read_expression_file(filepath):
    """
    智能读取表达矩阵文件，自动判断是否为 gzip 压缩格式。
    支持 .tsv, .txt, .tsv.gz, .txt.gz 等格式。
    """
    # 通过读取文件头 (magic number: 1f 8b) 判断是否为 gzip
    with open(filepath, 'rb') as f:
        is_gz = f.read(2) == b'\x1f\x8b'
    
    if is_gz:
        print(f"[INFO] Detected gzip compressed file: {os.path.basename(filepath)}")
        compression = 'gzip'
    else:
        # 如果不是 gz，让 pandas 根据后缀推断，或直接按纯文本处理
        compression = 'infer' 
        
    df = pd.read_csv(filepath, sep='\t', index_col=0, compression=compression)
    return df

def load_model(model_path, device, num_cell_types, mask_matrix):
    """加载模型并处理多GPU保存的 module. 前缀"""
    model = deconformer(num_cell_types, mask_matrix).to(device)
    
    # 根据设备类型决定 map_location，防止在 CPU 上加载 GPU 训练的模型时报错
    map_location = device if device.type != 'cpu' else torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=map_location)
    
    # 处理可能的 DataParallel 'module.' 前缀
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

def predict(model, data, device, num_cell_types, batch_size=256):
    model.eval()
    num_samples = data.X.shape[0]
    pre_frac = np.zeros((num_samples, num_cell_types))
    
    if hasattr(data.X, "toarray"):
        X_dense = data.X.toarray()
    else:
        X_dense = np.asarray(data.X)
        
    all_expressions = torch.tensor(X_dense, dtype=torch.float32)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Starting inference on {device} with batch_size={batch_size}...")
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_samples, batch_size), total=num_batches, desc="Inference"):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # 形状为 (current_batch_size, num_genes)
            batch_exp = all_expressions[start_idx:end_idx].to(device)
            
            # ================= 关键修复 =================
            # 增加一个维度，使其形状变为 (batch_size, 1, num_genes)
            # 这与训练时 DataLoader 提供的 3D 张量形状完全一致
            batch_exp = batch_exp.unsqueeze(1) 
            # ============================================
            
            predictions = model(batch_exp)
            pre_frac[start_idx:end_idx, :] = predictions.cpu().numpy()
            
    return pre_frac

def norm_real_data(df_data, gene):
    """数据标准化与重索引"""
    column_sums = df_data.sum()
    column_sums[column_sums == 0] = 1 
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
    """查找具体的模型 checkpoint 和 mask 文件"""
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

def run_inference(model_dir, epoch_str, cell_types_file, genes_file, exp_tsv, out_tsv, 
                  device, num_threads, batch_size):
    """执行推理的主流程"""
    
    # ================= 1. 设置设备与线程 =================
    if device.type == 'cpu':
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        print(f"[INFO] Using CPU with {num_threads} threads.")
    elif device.type == 'cuda':
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print(f"[INFO] Using Apple Silicon MPS backend.")
        
    # ================= 2. 加载基础配置 =================
    df_genes = pd.read_csv(genes_file, sep='\t', index_col=0)
    genes = df_genes.index.tolist()
    
    df_cell_types = pd.read_csv(cell_types_file, sep='\t', index_col=0)
    cell_type = df_cell_types.index.tolist()
    num_cell_types = len(cell_type)
    
    # ================= 3. 加载模型文件与 Mask =================
    model_pt_name, mask_name = get_mask_model(model_dir, epoch_str)
    print(f"[INFO] Using model checkpoint: {model_pt_name}")
    print(f"[INFO] Using mask file: {mask_name}")
    
    saved_model_path = os.path.join(model_dir, model_pt_name)
    mask_file_path = os.path.join(model_dir, mask_name)
    
    df_mask = pd.read_csv(mask_file_path, sep='\t', index_col=0)
    # 将 mask 矩阵直接加载到目标设备上
    mask_matrix = torch.from_numpy(df_mask.T.to_numpy()).int().to(device)
    
    # 4. 加载并预处理输入数据
    print(f"[INFO] Reading input file: {exp_tsv}")
    df_pred = read_expression_file(exp_tsv)  # <--- 使用智能读取函数
    ann_pred = norm_real_data(df_pred, genes)
    
    # ================= 修复 Warning =================
    # 加上 .copy() 避免在 view 上直接修改 X 触发 ImplicitModificationWarning
    ann_pred = ann_pred[:, df_mask.index.tolist()].copy()
    ann_pred.X = np.log2(ann_pred.X + 1)
    # =================================================
    
    # ================= 5. 加载模型 =================
    print("[INFO] Loading model...")
    loaded_model = load_model(saved_model_path, device, num_cell_types, mask_matrix)
    
    # ================= 6. 执行批量预测 =================
    print("[INFO] Starting inference...")
    t0 = time.time()
    pre_fraction = predict(loaded_model, ann_pred, device, num_cell_types, batch_size=batch_size)
    print(f"[INFO] Inference time: {time.time() - t0:.2f} seconds")
    
    # ================= 7. 保存结果 =================
    df_pre = pd.DataFrame(pre_fraction, index=ann_pred.obs_names.tolist(), columns=cell_type)
    
    out_dir = os.path.dirname(out_tsv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    df_pre.to_csv(out_tsv, sep='\t')
    print(f"[INFO] Inference complete. Results saved to: {out_tsv}")


# ================= 主入口 =================

def main():
    parser = argparse.ArgumentParser(
        description="Deconformer Prediction Tool (Optimized for CPU/GPU/MPS)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Available models:
  adult_model   : 60 basic cell types
  fetal_model   : 27 types of cells + 3 types of trophoblast cells + 4 types of fetal cells
  preg_model    : 60 types of cells + early and late stages of SCT, EVT, VCT, totaling six types of trophoblasts

Example usage:
  # CPU (threads 32, batch size 512)
  python deconformer_inference.py --model adult_model --input input.tsv --output output.tsv --device cpu --num-threads 32 --batch-size 512
  
  # GPU
  python deconformer_inference.py --model adult_model --input input.tsv --output output.tsv --device cuda
  
  # macOS MPS
  python deconformer_inference.py --model adult_model --input input.tsv --output output.tsv --device mps
        """
    )
    
    # 原有参数
    parser.add_argument('--model', '-m', type=str, required=True, 
                        help='Name of the trained model (e.g., adult_model, fetal_model, preg_model)')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to the input expression matrix TSV file')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Path to save the output inference result TSV file')
    
    # 新增参数：设备与并行控制
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device to run inference on: cpu (default), cuda (NVIDIA GPU), mps (Apple Silicon)')
    parser.add_argument('--num-threads', '-t', type=int, default=16,
                        help='Number of CPU threads to use when --device is cpu (default: 16)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Batch size for inference. Larger values utilize hardware better but consume more memory (default: 64)')
    
    args = parser.parse_args()
    
    # 验证模型名称
    if args.model not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        print(f"ERROR: '{args.model}' is not a valid model name.")
        print(f"Available models: {available}")
        sys.exit(1)
    
    # 解析并验证设备
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available on this machine. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    elif args.device == 'mps':
        if not torch.backends.mps.is_available():
            print("WARNING: MPS is not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = MODEL_CONFIGS[args.model]
    
    # 构建完整路径
    model_base_dir = os.path.join(script_dir, config['sub_dir'])
    cell_types_path = os.path.join(model_base_dir, config['cell_types_file'])
    genes_path = os.path.join(model_base_dir, config['genes_file'])
    
    # 检查必要文件是否存在
    for path, desc in [(model_base_dir, "Model directory"), (cell_types_path, "Cell types file"), 
                       (genes_path, "Genes file"), (args.input, "Input file")]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} not found: {path}")
            sys.exit(1)

    print(f"=== RUN: {args.model} on {device.type.upper()} {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    try:
        run_inference(
            model_dir=model_base_dir,
            epoch_str=config['epoch'],
            cell_types_file=cell_types_path,
            genes_file=genes_path,
            exp_tsv=args.input,
            out_tsv=args.output,
            device=device,
            num_threads=args.num_threads,
            batch_size=args.batch_size
        )
        print(f"=== DONE!!! {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    except Exception as e:
        print(f"=== ERROR: Inference failed. {str(e)} ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
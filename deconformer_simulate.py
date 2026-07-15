import os
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from numba import jit
import time
from joblib import Parallel, delayed
import argparse

# ================= 核心功能函数 =================

def load_data_from_h5ad(file_path, gene_order):
    print(f"Loading reference data from {file_path}...")
    ann = sc.read_h5ad(file_path)
    
    if 'cell_type' not in ann.obs.columns:
        raise ValueError("The 'cell_type' column is missing in ann.obs!")
        
    valid_genes = [g for g in gene_order if g in ann.var_names]
    if len(valid_genes) == 0:
        raise ValueError("No matching genes found in the h5ad file!")
        
    ann = ann[:, valid_genes]
    
    cell_data_dict = {}
    cell_types = ann.obs['cell_type'].unique()
    
    for ct in cell_types:
        ct_ann = ann[ann.obs['cell_type'] == ct, :]
        if isinstance(ct_ann.X, np.ndarray):
            cell_data_dict[ct] = ct_ann.X
        else:
            cell_data_dict[ct] = ct_ann.X.toarray()
            
    print(f"Successfully loaded data for {len(cell_data_dict)} cell types.")
    return cell_data_dict, valid_genes

@jit(nopython=True)
def generate_frac(num_all_cell_type):
    cell_composition = int(num_all_cell_type/3+0.5)
    sample_composition = np.random.randint(2, cell_composition)
    random_vector = np.random.random(size=(sample_composition,))
    vector_sum = np.sum(random_vector)
    normalized_vector = random_vector / vector_sum
    frac = np.zeros(num_all_cell_type)
    frac[:sample_composition] = normalized_vector
    np.random.shuffle(frac)
    return frac

def each_cell_contribution(cell_data, frac):
    n_cells = cell_data.shape[0]
    sample_counts = np.random.randint(200, 800)
    if sample_counts > n_cells:
        sample_counts = n_cells
    random_cells_idx = np.random.choice(n_cells, sample_counts, replace=False)
    exp = cell_data[random_cells_idx, :].mean(axis=0)
    contribute = exp * frac
    return contribute

# 1. 生成数据，直接返回 Numpy 数组，不包装 AnnData
def simulate_samples(cell_data_dict, cell_type_order, genes, batch, chunk_idx, n_samples=1000):
    X_chunk = np.zeros((n_samples, len(genes)), dtype=np.float32)
    Y_chunk = np.zeros((n_samples, len(cell_type_order)), dtype=np.float32)
    tags_chunk = []
    
    t0 = time.time()
    for sample_index in range(n_samples):
        tag = f"sample_{batch}_{chunk_idx}_{sample_index}"
        tags_chunk.append(tag)
        
        if sample_index % 500 == 0 and sample_index > 0:
            print(f"  Generated {sample_index}/{n_samples} in chunk {chunk_idx}, time: {time.time()-t0:.2f}s")
            
        frac = generate_frac(len(cell_data_dict))
        Y_chunk[sample_index, :] = frac
            
        sample_exp = np.zeros(len(genes), dtype=np.float32)
        for i, ct in enumerate(cell_type_order):
            f = frac[i]
            if f == 0:
                continue
            contribute = each_cell_contribution(cell_data_dict[ct], f)
            sample_exp += contribute
            
        X_chunk[sample_index, :] = sample_exp
        
    return X_chunk, Y_chunk, tags_chunk

# 2. 将 chunk 保存为轻量级的临时 HDF5 文件
def simulate_and_save_chunk(cell_data_dict, cell_type_order, genes, simu_batch, chunk_idx, n_samples, save_path):
    X_chunk, Y_chunk, tags_chunk = simulate_samples(cell_data_dict, cell_type_order, genes, simu_batch, chunk_idx, n_samples)
    
    chunk_save_path = os.path.join(save_path, f"tmp_batch{simu_batch}_chunk{chunk_idx}.h5")
    
    # 使用 h5py 直接写入，速度极快且无额外开销
    with h5py.File(chunk_save_path, 'w') as f:
        f.create_dataset('X', data=X_chunk, compression='gzip')
        f.create_dataset('Y', data=Y_chunk, compression='gzip')
        # 字符串类型需要使用特殊的 dtype
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('sample_ids', data=tags_chunk, dtype=dt)
        
    return chunk_save_path

def simulate_batch(cell_data_dict, cell_type_order, genes, simu_batch, n_sample, save_path, chunk_size):
    num_chunks = n_sample // chunk_size
    remainder = n_sample % chunk_size
    
    saved_files = []
    for i in range(num_chunks):
        print(f"[Batch {simu_batch}] Processing chunk {i+1}/{num_chunks}...")
        f = simulate_and_save_chunk(cell_data_dict, cell_type_order, genes, simu_batch, i, chunk_size, save_path)
        saved_files.append(f)
        
    if remainder > 0:
        print(f"[Batch {simu_batch}] Processing final chunk (remainder {remainder})...")
        f = simulate_and_save_chunk(cell_data_dict, cell_type_order, genes, simu_batch, num_chunks, remainder, save_path)
        saved_files.append(f)
        
    return saved_files

# ================= 命令行参数解析 =================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate cfRNA samples to pure HDF5 format for Deep Learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-i', '--input-h5ad', type=str, required=True, help='Path to the input single-cell reference .h5ad file.')
    parser.add_argument('-o', '--output-dir', type=str, default='./', help='Directory to save the final simulated .h5 file.')
    parser.add_argument('--final-name', type=str, default='simulated_data.h5', help='Name of the final merged h5 file.')
    parser.add_argument('--cell-types-file', type=str, required=True, help='Path to the cell types order file.')
    parser.add_argument('--genes-file', type=str, required=True, help='Path to the target genes list file.')
    
    parser.add_argument('-n', '--n-simulations', type=int, default=20, help='Number of parallel simulation batches.')
    parser.add_argument('-s', '--n-samples', type=int, default=20000, help='Total number of samples to generate per simulation batch.')
    parser.add_argument('-c', '--chunk-size', type=int, default=1000, help='Number of samples per chunk.')
    
    parser.add_argument('-j', '--jobs', type=int, default=None, help='Number of CPU cores to use.')
    parser.add_argument('--sep', type=str, default='\t', help='Separator for text files.')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary chunk files after merging.')
                        
    return parser.parse_args()

# ================= 主程序入口 =================

if __name__ == "__main__":
    args = parse_args()
    
    simulate_save_path = args.output_dir
    os.makedirs(simulate_save_path, exist_ok=True)

    df_cell_types = pd.read_csv(args.cell_types_file, sep=args.sep, index_col=0)
    cell_types_order = df_cell_types.index.tolist()
    genes = pd.read_csv(args.genes_file, sep=args.sep, index_col=0).index.tolist()

    n_simulations = args.n_simulations
    n_samples_per_simulation = args.n_samples
    chunk_size = args.chunk_size
    total_samples = n_simulations * n_samples_per_simulation
    
    n_jobs = args.jobs if args.jobs is not None else min(n_simulations, os.cpu_count() or 1)

    print("="*60)
    print(" Deconformer Simulation (Pure HDF5) ")
    print("="*60)
    print(f"  Total samples:    {total_samples}")
    print(f"  Total genes:      {len(genes)}")
    print(f"  Cell types:       {len(cell_types_order)}")
    print(f"  Final output:     {os.path.join(simulate_save_path, args.final_name)}")
    print("="*60)

    cell_data, valid_genes = load_data_from_h5ad(args.input_h5ad, genes)

    tp = time.time()
    
    # 1. 并行生成临时 chunk 文件
    print("\n[Step 1] Generating temporary chunks in parallel...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(simulate_batch)(cell_data, cell_types_order, valid_genes, simu_batch, n_samples_per_simulation, simulate_save_path, chunk_size)
        for simu_batch in range(n_simulations)
    )
    all_temp_files = [f for batch_files in results for f in batch_files]
    print(f"\n[Step 1] Finished. Generated {len(all_temp_files)} temporary files.")

    # 2. 【核心优化】零内存合并为最终 HDF5 文件
    print("\n[Step 2] Merging chunks into a single HDF5 file (Zero-Copy)...")
    final_file_path = os.path.join(simulate_save_path, args.final_name)
    
    with h5py.File(final_file_path, 'w') as f:
        # 创建可调整大小的数据集，并开启分块和压缩
        # chunks 参数决定了 HDF5 底层物理分块的大小，对后续 DL 随机读取至关重要
        dset_X = f.create_dataset('X', shape=(total_samples, len(valid_genes)), 
                                  maxshape=(None, len(valid_genes)),
                                  chunks=(chunk_size, len(valid_genes)), 
                                  compression='gzip', dtype='float32')
        
        dset_Y = f.create_dataset('Y', shape=(total_samples, len(cell_types_order)), 
                                  maxshape=(None, len(cell_types_order)),
                                  chunks=(chunk_size, len(cell_types_order)), 
                                  compression='gzip', dtype='float32')
        
        # 字符串数据集
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('genes', data=valid_genes, dtype=dt)
        f.create_dataset('cell_types', data=cell_types_order, dtype=dt)
        
        dset_sample_ids = f.create_dataset('sample_ids', shape=(total_samples,), 
                                           maxshape=(None,),
                                           chunks=(chunk_size,), 
                                           compression='gzip', dtype=dt)

        # 串行读取临时文件，直接按 offset 写入磁盘，不占用额外内存
        offset = 0
        for i, temp_file in enumerate(all_temp_files):
            print(f"  Appending file {i+1}/{len(all_temp_files)}: {os.path.basename(temp_file)}")
            with h5py.File(temp_file, 'r') as tf:
                current_chunk_size = tf['X'].shape[0]
                # 直接写入磁盘对应位置
                dset_X[offset:offset+current_chunk_size, :] = tf['X'][:]
                dset_Y[offset:offset+current_chunk_size, :] = tf['Y'][:]
                dset_sample_ids[offset:offset+current_chunk_size] = tf['sample_ids'][:]
                offset += current_chunk_size
                
    print(f"  -> Successfully saved to {final_file_path}")

    # 3. 清理临时文件
    if not args.keep_temp:
        print("\n[Step 3] Cleaning up temporary files...")
        for f in all_temp_files:
            os.remove(f)
        print("  -> Temporary files deleted.")

    print(f"\nTotal time: {time.time()-tp:.2f}s")
    print("Done!")
import time
import pandas as pd
import numpy as np
import random
import os
import gc
import h5py

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== Flash Attention 检测 =====================

def check_flash_attention_availability():
    """检测 Flash Attention 的可用性并打印信息"""
    print("\n" + "="*60)
    print(" Attention Backend Detection ")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available.")
        print("="*60 + "\n")
        return
    
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # try:
    #     import flash_attn
    #     print(f"✅ flash-attn package installed (version: {flash_attn.__version__})")
    # except ImportError:
    #     print("⚠️  flash-attn package NOT installed")
    
    print("="*60 + "\n")

check_flash_attention_availability()


# ===================== 数据工具函数 =====================

def compute_gene_variance_h5(h5_path, chunk_size=50000):
    """从 HDF5 文件流式计算每个基因的方差"""
    with h5py.File(h5_path, 'r') as f:
        n_samples, n_genes = f['X'].shape
        sum_x = np.zeros(n_genes, dtype=np.float64)
        sum_x2 = np.zeros(n_genes, dtype=np.float64)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = f['X'][start:end, :].astype(np.float64)
            row_sums = chunk.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            chunk = chunk / row_sums * 10000.0
            sum_x += chunk.sum(axis=0)
            sum_x2 += (chunk ** 2).sum(axis=0)

        mean = sum_x / n_samples
        var = sum_x2 / n_samples - mean ** 2
    return var


def top_variance_gene_h5(h5_path, n_topgenes, outdir):
    """从 HDF5 文件选取高方差基因"""
    var = compute_gene_variance_h5(h5_path)
    with h5py.File(h5_path, 'r') as f:
        genes = [g.decode('utf-8') if isinstance(g, bytes) else g for g in f['genes'][:]]

    df_var = pd.DataFrame({'variance': var}, index=genes)
    df_sorted = df_var.sort_values(by='variance', ascending=False)
    df_sorted = df_sorted[df_sorted['variance'] > 0]
    df_vartop = df_sorted.head(n_topgenes)
    print("Top variance genes:")
    print(df_vartop.head(20))
    df_vartop.to_csv(os.path.join(outdir, f"top_variance_genes_{df_vartop.shape[0]}.txt"), sep='\t')
    return df_vartop


def read_pathway_gmtfile(file, tvg, th):
    """读取 GMT 通路文件"""
    with open(file, 'r') as f:
        genesets = {}
        genesets_rate = {}
        for line in f:
            split_line = line.strip().split('\t')
            geneset_name = split_line[0]
            genes = split_line[2:]
            rate = len(set(genes).intersection(set(tvg))) / (len(set(genes).union(set(tvg))) + 1e-9)
            genesets_rate[geneset_name] = rate
            genesets[geneset_name] = genes
        rate_top = sorted(genesets_rate.items(), key=lambda item: item[1], reverse=True)[:th]
        result = [item[0] for item in rate_top]
        top_pathway = {key: genesets[key] for key in result}
    return genesets_rate, top_pathway


def pathway_mask(pathways, tvg, outdir):
    """构建通路-基因 mask 矩阵"""
    gene = list(pathways.values())
    flattened_list = [item for sublist in gene for item in sublist]
    uni_list = list(set(flattened_list))
    print(f"Pathway genes count: {len(uni_list)}")

    inter_tvg = list(set(tvg).intersection(set(uni_list)))
    random.shuffle(inter_tvg)
    print(f"Intersection TVG-pathway genes count: {len(inter_tvg)}")

    matrix = pd.DataFrame(
        np.zeros((len(inter_tvg), len(pathways))),
        index=inter_tvg,
        columns=list(pathways.keys())
    )
    for pathway, gene_list in pathways.items():
        inter_gene = list(set(inter_tvg).intersection(set(gene_list)))
        matrix.loc[inter_gene, pathway] = 1
    matrix.to_csv(
        os.path.join(outdir, f"mask_gene_{matrix.shape[1]}_pathway{matrix.shape[0]}.txt"),
        sep='\t'
    )
    return matrix


# ===================== 模型定义（优化显存版本） =====================

class deconformer(nn.Module):
    def __init__(self, num_cell_types, mask):
        super(deconformer, self).__init__()

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
            activation='gelu',
            num_decoder_layers=0,
            dropout=0.1
        )

        self.fraction = nn.Linear(self.embedding_dim, num_cell_types)

    def forward(self, x):
        """
        优化版本：使用 einsum 避免创建大张量
        x: (batch, n_genes)
        """
        # 使用 einsum 直接计算，避免创建 (batch, n_pathways, n_genes) 的大张量
        # result[b,p,e] = sum_g(x[b,g] * mask[p,g] * embedding_matrix[g,e])
        x = torch.einsum('bg,pg,ge->bpe', x, self.mask.to(x.device), self.embedding_matrix)
        
        # 拼接 CLS token
        cls_token = self.cls_token_embedding.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # nn.Transformer 需要 (seq_len, batch, dim) 格式
        x = x.permute(1, 0, 2)
        x = self.transformer.encoder(x)
        
        # 取 CLS token 的输出
        cls_out = x[0]
        output = self.fraction(cls_out)
        softmax_output = torch.softmax(output, dim=1)
        return softmax_output


# ===================== HDF5 Dataset =====================

class SimuH5Dataset(Dataset):
    """从 HDF5 文件惰性读取模拟数据"""
    def __init__(self, h5_path, sample_indices, gene_indices,
                 training=False, dropout_rate=0.0):
        self.h5_path = h5_path
        self.sample_indices = np.asarray(sample_indices)
        self.gene_indices = np.asarray(gene_indices)
        self.training = training
        self.dropout_rate = dropout_rate
        self.h5_file = None

        # 排序 gene_indices 以兼容 h5py fancy indexing
        self._gene_sort_order = np.argsort(self.gene_indices)
        self._sorted_gene_idx = self.gene_indices[self._gene_sort_order]
        self._gene_unsort = np.argsort(self._gene_sort_order)

        # 排序 sample_indices 读取 Y
        sample_sort_order = np.argsort(self.sample_indices)
        sorted_samples = self.sample_indices[sample_sort_order]

        with h5py.File(h5_path, 'r') as f:
            sorted_labels = f['Y'][sorted_samples]

        sample_unsort = np.argsort(sample_sort_order)
        self.labels = sorted_labels[sample_unsort]

    def __len__(self):
        return len(self.sample_indices)

    def _ensure_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        h5_idx = int(self.sample_indices[idx])

        expression = self.h5_file['X'][h5_idx, self._sorted_gene_idx].astype(np.float32)
        expression = expression[self._gene_unsort]

        # 增加 CPM 归一化（与老代码的 normalize_total 一致）
        row_sum = expression.sum()
        if row_sum > 0:
            expression = expression / row_sum * 10000.0

        expression = np.log2(expression + 1.0)

        if self.training and self.dropout_rate > 0:
            drop_mask = np.random.random(len(expression)) > self.dropout_rate
            expression = expression * drop_mask

        expression = torch.from_numpy(expression)
        label = torch.from_numpy(self.labels[idx].astype(np.float32))
        return expression, label

    def __del__(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass


# ===================== 损失函数 =====================

def mse_non_zero(y_pred, y_true):
    non_zero = torch.nonzero(y_true, as_tuple=True)
    if non_zero[0].numel() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    return torch.mean((y_true[non_zero] - y_pred[non_zero]) ** 2)


def mae_non_zero(y_pred, y_true):
    non_zero = torch.nonzero(y_true, as_tuple=True)
    if non_zero[0].numel() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    return torch.mean(torch.abs(y_true[non_zero] - y_pred[non_zero]))


# ===================== 学习率调度器 =====================

class WarmupConstantLRScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            lr = self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr


# ===================== 训练与评估 =====================

def evaluate(model, dataloader, loss_fn_name):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for expressions, frac in dataloader:
            expressions = expressions.to(device)
            frac = frac.to(device)
            outputs = model(expressions)
            loss = mae_non_zero(outputs, frac) if loss_fn_name == "MAE" else mse_non_zero(outputs, frac)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def train(model, loss_function, total_epoch, save_path, optimizer,
          dataloader, test_dataloader, scheduler):
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(total_epoch):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for expressions, frac in pbar:
            expressions = expressions.to(device)
            frac = frac.to(device)

            current_lr = scheduler.step()
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(expressions)
                loss = mae_non_zero(outputs, frac) if loss_function == "MAE" else mse_non_zero(outputs, frac)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            batch_count += 1
            pbar.set_description(
                f"Epoch {epoch+1} avgLoss: {loss_sum/batch_count:.5f}, "
                f"Loss: {loss.item():.5f}, LR: {current_lr:.6f}"
            )

        test_avg_loss = evaluate(model, test_dataloader, loss_function)
        print(f"Epoch {epoch+1}, Test Avg Loss: {test_avg_loss:.5f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(save_path, f"model_checkpoint_epoch_{epoch+1}.pt"))


# ===================== 主流程 =====================

def fitting_pipelines(project, h5_path, mask_file, gmtfile,
                      loss="MSE", learning_rate=0.0005, n_pathway=5000,
                      batch_size=128, epoch=15, n_tvg=10000,
                      dropout_rate=0.2, num_workers=4, train_ratio=0.9975):
    run_time = time.strftime("%m%d_%H%M%S")
    project = project or "project_"
    project = project + str(run_time)
    print(f"Project name: {project}")
    project_path = os.path.join(os.getcwd(), project)
    os.makedirs(project_path, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        h5_genes = [g.decode('utf-8') if isinstance(g, bytes) else g for g in f['genes'][:]]
        h5_cell_types = [c.decode('utf-8') if isinstance(c, bytes) else c for c in f['cell_types'][:]]
        n_samples_total = f['X'].shape[0]
    print(f"HDF5 file: {n_samples_total} samples, {len(h5_genes)} genes, {len(h5_cell_types)} cell types")

    print("Selecting top variance genes...")
    tvg_df = top_variance_gene_h5(h5_path, n_tvg, project_path)
    tvg = tvg_df.index.tolist()

    print("Reading pathway GMT file...")
    _, top_pathway = read_pathway_gmtfile(gmtfile, tvg, n_pathway)

    if mask_file is not None:
        df_mask = pd.read_csv(mask_file, sep='\t', index_col=0)
        print(f"Loaded pre-computed mask: {df_mask.shape}")
    else:
        df_mask = pathway_mask(top_pathway, tvg, project_path)

    num_cell_types = len(h5_cell_types)
    mask = df_mask.T.to_numpy()
    input_genes = df_mask.index.tolist()
    mask_matrix = torch.from_numpy(mask).int()
    print(f"Mask tensor shape: {mask_matrix.shape}")

    gene_to_h5idx = {g: i for i, g in enumerate(h5_genes)}
    gene_indices = np.array([gene_to_h5idx[g] for g in input_genes if g in gene_to_h5idx])
    print(f"Gene indices mapped: {len(gene_indices)} genes")

    np.random.seed(0)
    indices = np.random.permutation(n_samples_total)
    n_train = int(n_samples_total * train_ratio)
    indices_train = indices[:n_train]
    indices_test = indices[n_train:]
    print(f"Train: {len(indices_train)}, Test: {len(indices_test)}")

    train_dataset = SimuH5Dataset(h5_path, indices_train, gene_indices, training=True, dropout_rate=dropout_rate)
    test_dataset = SimuH5Dataset(h5_path, indices_test, gene_indices, training=False, dropout_rate=0.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                             prefetch_factor=4, persistent_workers=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)

    print("Building model...")
    model = deconformer(num_cell_types, mask_matrix)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = WarmupConstantLRScheduler(optimizer, warmup_steps=10000, max_lr=learning_rate)

    t0 = time.time()
    train(model=model, loss_function=loss, total_epoch=epoch, save_path=project_path,
          optimizer=optimizer, dataloader=train_loader, test_dataloader=test_loader, scheduler=scheduler)
    print(f"Total training time: {time.time()-t0:.1f}s")
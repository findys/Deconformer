import argparse
import sys
from deconformer_model import fitting_pipelines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Deconformer model from simulated HDF5 data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument('-i', '--input-h5', type=str, required=True,
                        help='Path to the simulated HDF5 file (from deconformer_simulate.py).')
    parser.add_argument('--gmt', type=str, required=True,
                        help='Pathway GMT file (e.g., c5.go.bp.v2023.1.Hs.symbols.gmt).')
    parser.add_argument('--project-name', type=str, required=True,
                        help='Project name prefix for output directory.')

    # 可选参数
    parser.add_argument('--mask-file', type=str, default=None,
                        help='Pre-computed pathway mask file (tsv). If None, computed from data.')
    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'MAE'],
                        help='Loss function.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--n-pathways', type=int, default=5000,
                        help='Number of top pathways to use.')
    parser.add_argument('--n-tvg', type=int, default=10000,
                        help='Number of top variance genes to select.')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Gene dropout rate for data augmentation (training only).')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of DataLoader workers for HDF5 reading.')
    parser.add_argument('--train-ratio', type=float, default=0.9975,
                        help='Ratio of samples used for training (rest for test).')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print(" Deconformer Training (HDF5) ")
    print("=" * 60)
    print(f"  Input HDF5:     {args.input_h5}")
    print(f"  GMT file:       {args.gmt}")
    print(f"  Project:        {args.project_name}")
    print(f"  Loss:           {args.loss}")
    print(f"  LR:             {args.lr}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  N pathways:     {args.n_pathways}")
    print(f"  N TVG:          {args.n_tvg}")
    print(f"  Dropout rate:   {args.dropout_rate}")
    print(f"  Num workers:    {args.num_workers}")
    print("=" * 60)

    fitting_pipelines(
        project=args.project_name,
        h5_path=args.input_h5,
        mask_file=args.mask_file,
        gmtfile=args.gmt,
        loss=args.loss,
        learning_rate=args.lr,
        n_pathway=args.n_pathways,
        batch_size=args.batch_size,
        epoch=args.epochs,
        n_tvg=args.n_tvg,
        dropout_rate=args.dropout_rate,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
    )
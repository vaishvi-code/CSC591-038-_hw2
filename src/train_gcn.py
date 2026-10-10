import patch_torch
import argparse
import os
import time
import json
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import GCN


def parse_args():
    parser = argparse.ArgumentParser(description='Task 1: Full-Batch GCN on ogbn-arxiv')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3, help='Model depth (layers)')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./results/metrics')
    parser.add_argument('--exp_name', type=str, default='gcn_baseline')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"=== Running GCN (Depth: {args.num_layers}, LR: {args.lr}, Hidden: {args.hidden_channels}, Device: {device}) ===")

    # Load dataset
    # To avoid requiring torch-sparse C++ binaries, we can use edge_index directly or ToSparseTensor if available
    try:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.dataset_dir,
                                         transform=T.ToSparseTensor())
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        graph_repr = data.adj_t
    except Exception as e:
        print(f"SparseTensor conversion failed ({e}), falling back to standard edge_index.")
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.dataset_dir)
        data = dataset[0]
        from torch_geometric.utils import to_undirected
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        graph_repr = data.edge_index

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    val_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    x = data.x.to(device)
    y = data.y.squeeze(1).to(device)
    graph_repr = graph_repr.to(device)

    model = GCN(in_channels=data.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'epoch_times': [],
        'peak_memory_mb': 0.0,
        'config': vars(args)
    }

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        optimizer.zero_grad()
        out = model(x, graph_repr)
        train_loss = F.nll_loss(out[train_idx], y[train_idx])
        train_loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(x, graph_repr)
            val_loss = F.nll_loss(out[val_idx], y[val_idx]).item()
            test_loss = F.nll_loss(out[test_idx], y[test_idx]).item()

            y_pred = out.argmax(dim=-1, keepdim=True)
            train_acc = evaluator.eval({
                'y_true': data.y[split_idx['train']],
                'y_pred': y_pred[split_idx['train']].cpu(),
            })['acc']
            val_acc = evaluator.eval({
                'y_true': data.y[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']].cpu(),
            })['acc']
            test_acc = evaluator.eval({
                'y_true': data.y[split_idx['test']],
                'y_pred': y_pred[split_idx['test']].cpu(),
            })['acc']

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch: {epoch:03d} | Time: {epoch_time:.3f}s | "
                  f"Loss: Train {train_loss.item():.4f}, Val {val_loss:.4f}, Test {test_loss:.4f} | "
                  f"Acc: Train {train_acc*100:.2f}%, Val {val_acc*100:.2f}%, Test {test_acc*100:.2f}%")

    total_time = time.time() - total_start_time

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        import psutil
        peak_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    history['peak_memory_mb'] = peak_mem
    history['total_time_sec'] = total_time
    history['avg_epoch_time_sec'] = sum(history['epoch_times']) / len(history['epoch_times'])
    history['best_val_acc'] = max(history['val_acc'])
    history['best_test_acc'] = history['test_acc'][history['val_acc'].index(history['best_val_acc'])]

    print(f"\nCompleted in {total_time:.2f}s | Peak Memory: {peak_mem:.2f} MB")
    print(f"Best Validation Accuracy: {history['best_val_acc']*100:.2f}% (Test: {history['best_test_acc']*100:.2f}%)")

    # Save metrics JSON
    output_path = os.path.join(args.output_dir, f"{args.exp_name}.json")
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved run metrics to {output_path}")


if __name__ == '__main__':
    main()

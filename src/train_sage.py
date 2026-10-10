import argparse
import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler

from models import GraphSAGE


def parse_args():
    parser = argparse.ArgumentParser(description='Task 2: Mini-Batch GraphSAGE on ogbn-products')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3, help='Model depth')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size')
    parser.add_argument('--sizes', type=str, default='15,10,5', help='Neighbor sample sizes per layer, comma-separated')
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./results/metrics')
    parser.add_argument('--exp_name', type=str, default='sage_baseline')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Parse sampling sizes
    sample_sizes = [int(s.strip()) for s in args.sizes.split(',')]
    assert len(sample_sizes) == args.num_layers, f"Number of sample sizes ({len(sample_sizes)}) must match num_layers ({args.num_layers})"

    print(f"=== Running GraphSAGE (Layers: {args.num_layers}, Fanout: {sample_sizes}, Batch: {args.batch_size}, LR: {args.lr}, Device: {device}) ===")

    # Load dataset
    dataset = PygNodePropPredDataset(name='ogbn-products', root=args.dataset_dir)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    train_loader = NeighborSampler(
        data.edge_index, node_idx=train_idx,
        sizes=sample_sizes, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    # Subgraph loader for layer-by-layer evaluation
    subgraph_loader = NeighborSampler(
        data.edge_index, node_idx=None, sizes=[-1],
        batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    evaluator = Evaluator(name='ogbn-products')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    x = data.x
    y = data.y.squeeze(1).to(device)

    history = {
        'epoch': [],
        'train_loss': [],
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
        model.train()
        total_loss = 0
        total_correct = 0
        total_train_nodes = 0

        for batch_size, n_id, adjs in train_loader:
            # Transfer batch subgraph data to device
            adjs = [adj.to(device) for adj in adjs]
            batch_x = x[n_id].to(device)
            batch_y = y[n_id[:batch_size]]

            optimizer.zero_grad()
            out = model(batch_x, adjs)
            loss = F.nll_loss(out, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_correct += int((out.argmax(dim=-1) == batch_y).sum())
            total_train_nodes += batch_size

        train_loss = total_loss / total_train_nodes
        train_acc = total_correct / total_train_nodes
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)

        # Evaluation via layer-by-layer inference
        model.eval()
        with torch.no_grad():
            out_all = model.inference(x, subgraph_loader, device)
            y_pred = out_all.argmax(dim=-1, keepdim=True).cpu()

            val_acc = evaluator.eval({
                'y_true': data.y[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': data.y[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['acc']

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch: {epoch:02d} | Time: {epoch_time:.2f}s | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

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

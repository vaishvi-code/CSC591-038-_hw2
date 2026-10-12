import patch_torch
import argparse
import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import GraphSAGE
from sampler import FastTorchNeighborSampler


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
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./results/metrics')
    parser.add_argument('--exp_name', type=str, default='sage_baseline')
    return parser.parse_args()


def iterate_batches(indices, batch_size, shuffle=True):
    if shuffle:
        perm = torch.randperm(len(indices), device=indices.device)
        indices = indices[perm]
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]


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

    train_idx = split_idx['train'].to(device)
    val_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    x = data.x.to(device)
    y = data.y.squeeze(1).to(device)

    print("Building high-speed GPU neighbor sampler index...")
    sampler = FastTorchNeighborSampler(data.edge_index, data.num_nodes, device=device)
    print("Graph sampler ready!")

    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
    best_val_acc = 0.0
    best_test_acc = 0.0
    train_peak_mb = 0.0  # training-only peak, snapshotted before eval each epoch

    @torch.no_grad()
    def evaluate(eval_indices, batch_size=4096):
        model.eval()
        correct = 0
        total = len(eval_indices)
        for batch_seeds in iterate_batches(eval_indices, batch_size, shuffle=False):
            ordered_nodes, sub_edge_index = sampler.sample(batch_seeds, sample_sizes)
            batch_x = x[ordered_nodes]
            batch_y = y[batch_seeds]

            out = model(batch_x, sub_edge_index)
            pred = out[:len(batch_seeds)].argmax(dim=-1)
            correct += int((pred == batch_y).sum())
        return correct / total if total > 0 else 0.0

    num_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_train_nodes = 0

        pbar = tqdm(iterate_batches(train_idx, args.batch_size, shuffle=True),
                    total=num_batches, desc=f"Epoch {epoch:02d}")

        for batch_seeds in pbar:
            ordered_nodes, sub_edge_index = sampler.sample(batch_seeds, sample_sizes)
            batch_x = x[ordered_nodes]
            batch_y = y[batch_seeds]

            optimizer.zero_grad()
            out = model(batch_x, sub_edge_index)
            target_out = out[:len(batch_seeds)]

            loss = F.nll_loss(target_out, batch_y)
            loss.backward()
            optimizer.step()

            bs = len(batch_seeds)
            total_loss += loss.item() * bs
            total_correct += int((target_out.argmax(dim=-1) == batch_y).sum())
            total_train_nodes += bs

            pbar.set_postfix({
                'loss': f"{total_loss / total_train_nodes:.4f}",
                'train_acc': f"{(total_correct / total_train_nodes) * 100:.2f}%"
            })

        train_loss = total_loss / total_train_nodes
        train_acc = total_correct / total_train_nodes
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)

        # Snapshot training-only CUDA peak BEFORE the eval pass inflates it.
        # max_memory_allocated is monotonically increasing (reset once at start),
        # so this correctly captures the peak reached during training.
        if device.type == 'cuda':
            cur_train_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if cur_train_peak > train_peak_mb:
                train_peak_mb = cur_train_peak

        # Evaluate validation
        val_acc = evaluate(val_idx, batch_size=args.eval_batch_size)

        # Evaluate test if best val or last epoch
        if val_acc > best_val_acc or epoch == args.epochs or epoch == 1:
            test_acc = evaluate(test_idx, batch_size=args.eval_batch_size)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
        else:
            test_acc = best_test_acc

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch:02d} Summary: Time {epoch_time:.2f}s | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    total_time = time.time() - total_start_time

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        import psutil
        peak_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    history['peak_memory_mb'] = peak_mem            # global peak (includes eval pass)
    history['eval_peak_memory_mb'] = peak_mem        # alias for clarity
    history['train_peak_memory_mb'] = train_peak_mb  # training-only peak (before eval)
    history['total_time_sec'] = total_time
    history['avg_epoch_time_sec'] = sum(history['epoch_times']) / len(history['epoch_times'])
    history['best_val_acc'] = best_val_acc
    history['best_test_acc'] = best_test_acc

    print(f"\nCompleted in {total_time:.2f}s | Peak Memory: {peak_mem:.2f} MB")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Test: {best_test_acc*100:.2f}%)")

    # Save metrics JSON
    output_path = os.path.join(args.output_dir, f"{args.exp_name}.json")
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved run metrics to {output_path}")


if __name__ == '__main__':
    main()

import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_gcn_convergence(metrics_dir, output_dir):
    baseline_path = os.path.join(metrics_dir, "gcn_baseline.json")
    if not os.path.exists(baseline_path):
        print(f"File {baseline_path} not found, skipping GCN convergence plot.")
        return

    with open(baseline_path, 'r') as f:
        data = json.load(f)

    epochs = data['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, data['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, data['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    ax1.plot(epochs, data['test_loss'], label='Test Loss', color='#2ca02c', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('GCN Convergence: Cross-Entropy Loss')
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in data['train_acc']], label='Train Acc', color='#1f77b4', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in data['val_acc']], label='Val Acc', color='#ff7f0e', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in data['test_acc']], label='Test Acc', color='#2ca02c', linestyle='--', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('GCN Convergence: Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "gcn_convergence.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_gcn_ablations(metrics_dir, output_dir):
    depth_files = glob.glob(os.path.join(metrics_dir, "gcn_depth_*.json"))
    lr_files = glob.glob(os.path.join(metrics_dir, "gcn_lr_*.json"))

    # Depth ablation
    if depth_files:
        depths, val_accs, test_accs = [], [], []
        for p in sorted(depth_files):
            with open(p, 'r') as f:
                d = json.load(f)
                depths.append(d['config']['num_layers'])
                val_accs.append(d['best_val_acc'] * 100)
                test_accs.append(d['best_test_acc'] * 100)

        # Sort by depth
        sorted_pairs = sorted(zip(depths, val_accs, test_accs))
        depths, val_accs, test_accs = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(depths, val_accs, marker='o', label='Best Validation Acc', color='#ff7f0e', linewidth=2)
        ax.plot(depths, test_accs, marker='s', label='Test Acc', color='#1f77b4', linewidth=2)
        ax.set_xlabel('Model Depth (Number of Layers)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('GCN Performance vs. Model Depth')
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, "gcn_depth_ablation.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    # LR ablation
    if lr_files:
        lrs, val_accs, test_accs = [], [], []
        for p in sorted(lr_files):
            with open(p, 'r') as f:
                d = json.load(f)
                lrs.append(d['config']['lr'])
                val_accs.append(d['best_val_acc'] * 100)
                test_accs.append(d['best_test_acc'] * 100)

        sorted_pairs = sorted(zip(lrs, val_accs, test_accs))
        lrs, val_accs, test_accs = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(lrs, val_accs, marker='o', label='Best Validation Acc', color='#ff7f0e', linewidth=2)
        ax.plot(lrs, test_accs, marker='s', label='Test Acc', color='#1f77b4', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('GCN Performance vs. Learning Rate')
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, "gcn_lr_ablation.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")


def plot_sage_results(metrics_dir, output_dir):
    baseline_path = os.path.join(metrics_dir, "sage_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            data = json.load(f)

        epochs = data['epoch']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, data['train_loss'], label='Train Loss', color='#d62728', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('GraphSAGE Convergence: Cross-Entropy Loss')
        ax1.legend()

        ax2.plot(epochs, [a * 100 for a in data['train_acc']], label='Train Acc', color='#1f77b4', linewidth=2)
        ax2.plot(epochs, [a * 100 for a in data['val_acc']], label='Val Acc', color='#ff7f0e', linewidth=2)
        ax2.plot(epochs, [a * 100 for a in data['test_acc']], label='Test Acc', color='#2ca02c', linestyle='--', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('GraphSAGE Accuracy (%) over Epochs')
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(output_dir, "sage_convergence.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    # SAGE Batch Size / Sampling Ablations
    batch_files = glob.glob(os.path.join(metrics_dir, "sage_batch_*.json"))
    if batch_files:
        batches, val_accs, test_accs = [], [], []
        for p in sorted(batch_files):
            with open(p, 'r') as f:
                d = json.load(f)
                batches.append(d['config']['batch_size'])
                val_accs.append(d['best_val_acc'] * 100)
                test_accs.append(d['best_test_acc'] * 100)

        sorted_pairs = sorted(zip(batches, val_accs, test_accs))
        batches, val_accs, test_accs = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([str(b) for b in batches], val_accs, marker='o', label='Val Acc', color='#ff7f0e', linewidth=2)
        ax.plot([str(b) for b in batches], test_accs, marker='s', label='Test Acc', color='#1f77b4', linewidth=2)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('GraphSAGE Performance vs. Batch Size')
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, "sage_batch_ablation.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")


def plot_comparative_profiling(metrics_dir, output_dir):
    gcn_path = os.path.join(metrics_dir, "gcn_baseline.json")
    sage_path = os.path.join(metrics_dir, "sage_baseline.json")

    if not (os.path.exists(gcn_path) and os.path.exists(sage_path)):
        print("Both gcn_baseline.json and sage_baseline.json are needed for comparative profiling.")
        return

    with open(gcn_path, 'r') as f:
        gcn = json.load(f)
    with open(sage_path, 'r') as f:
        sage = json.load(f)

    models = ['Full-Batch GCN\n(ogbn-arxiv)', 'Mini-Batch GraphSAGE\n(ogbn-products)']
    peak_memory = [gcn['peak_memory_mb'], sage['peak_memory_mb']]
    epoch_times = [gcn['avg_epoch_time_sec'], sage['avg_epoch_time_sec']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Memory comparison
    bars1 = ax1.bar(models, peak_memory, color=['#1f77b4', '#2ca02c'], width=0.5)
    ax1.set_ylabel('Peak Memory Consumption (MB)')
    ax1.set_title('Memory Scalability Comparison')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval:.1f} MB", ha='center', va='bottom', fontweight='bold')

    # Runtime comparison
    bars2 = ax2.bar(models, epoch_times, color=['#ff7f0e', '#d62728'], width=0.5)
    ax2.set_ylabel('Average Runtime per Epoch (s)')
    ax2.set_title('Training Speed Comparison')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.2f} s", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "comparative_profiling.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    metrics_dir = "./results/metrics"
    output_dir = "./results/plots"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots from metrics...")
    plot_gcn_convergence(metrics_dir, output_dir)
    plot_gcn_ablations(metrics_dir, output_dir)
    plot_sage_results(metrics_dir, output_dir)
    plot_comparative_profiling(metrics_dir, output_dir)
    print("Done generating plots.")


if __name__ == '__main__':
    main()

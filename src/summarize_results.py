import glob
import json

print("="*65)
print(f"{'Experiment':30s} | {'Config':15s} | {'Val Acc':8s} | {'Test Acc':8s} | {'Memory':9s} | {'Epoch Time'}")
print("="*65)

print("\n--- Task 1: GCN on ogbn-arxiv ---")
for f in sorted(glob.glob("results/metrics/gcn_*.json")):
    with open(f) as fp:
        d = json.load(fp)
    cfg = d["config"]
    name = f.replace("results/metrics\\", "").replace("results/metrics/", "").replace(".json", "")
    cfg_str = f"L={cfg.get('num_layers')}, lr={cfg.get('lr')}"
    val_acc = f"{d['best_val_acc']*100:.2f}%"
    test_acc = f"{d['best_test_acc']*100:.2f}%"
    mem = f"{d['peak_memory_mb']:.1f} MB"
    ep_time = f"{d['avg_epoch_time_sec']:.4f} s"
    print(f"{name:30s} | {cfg_str:15s} | {val_acc:8s} | {test_acc:8s} | {mem:9s} | {ep_time}")

print("\n--- Task 2: GraphSAGE on ogbn-products ---")
for f in sorted(glob.glob("results/metrics/sage_*.json")):
    with open(f) as fp:
        d = json.load(fp)
    cfg = d["config"]
    name = f.replace("results/metrics\\", "").replace("results/metrics/", "").replace(".json", "")
    cfg_str = f"B={cfg.get('batch_size')}, S={cfg.get('sizes')}"
    val_acc = f"{d['best_val_acc']*100:.2f}%"
    test_acc = f"{d['best_test_acc']*100:.2f}%"
    mem = f"{d['peak_memory_mb']:.1f} MB"
    ep_time = f"{d['avg_epoch_time_sec']:.2f} s"
    print(f"{name:30s} | {cfg_str:15s} | {val_acc:8s} | {test_acc:8s} | {mem:9s} | {ep_time}")

print("="*65)

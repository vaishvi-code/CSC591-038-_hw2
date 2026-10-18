# CSC 591/791 Homework 2: GCN vs. GraphSAGE & Scalability Analysis

This repository contains the complete implementation, experimental pipelines, and reproduction scripts for comparing full-batch Graph Convolutional Networks (GCN) and mini-batch GraphSAGE on Open Graph Benchmark (OGB) datasets.

---

## 📁 Repository Structure

```
CSC591-038-_hw2/
├── .gitignore
├── requirements.txt
├── README.md
├── notebooks/
│   └── HW2_GCN_GraphSAGE_Experiments.ipynb  # End-to-end runnable Google Colab Notebook
├── src/
│   ├── models.py                            # Modular GCN and GraphSAGE model architectures
│   ├── train_gcn.py                         # Task 1: Full-batch GCN on ogbn-arxiv
│   ├── train_sage.py                        # Task 2: Mini-batch GraphSAGE on ogbn-products
│   ├── plot_results.py                      # Task 3: Publication-ready plotting & profiling charts
│   ├── logger.py                            # OGB evaluator logger utility
│   ├── reference_arxiv_gnn.py               # Official OGB reference implementation for ogbn-arxiv
│   ├── reference_products_gnn.py            # Official OGB reference implementation for ogbn-products
│   └── reference_pyg_sage.py                # Official PyG reference implementation with NeighborSampler
├── results/
│   ├── metrics/                             # JSON logs of runs (loss, accuracy, memory, latency)
│   └── plots/                               # Generated high-resolution PNG figures for report
└── report/                                  # 3-page written PDF report source & documents
```

---

## 🚀 Running on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Open Notebook > GitHub** and enter:
   ```
   https://github.com/vaishvi-code/CSC591-038-_hw2
   ```
3. Open `notebooks/HW2_GCN_GraphSAGE_Experiments.ipynb`.
4. Change runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
5. Run the cells step-by-step:
   - Clones repo & installs dependencies (`torch-geometric`, `ogb`).
   - Runs Task 1: GCN baseline + depth ablation + learning rate ablation.
   - Runs Task 2: GraphSAGE baseline + batch size ablation + neighbor sampling ablation.
   - Runs Task 3: Generates all comparison plots in `results/plots/`.
   - Downloads `results.zip`.

---

## 💻 Running Locally / On Cluster

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Task 1: Full-Batch GCN on `ogbn-arxiv` (Table 1, Figures 1 & 2)
```bash
# Baseline GCN Reference (Depth 3, LR 0.01, 100 epochs) -> Fig 1, Table 1
python src/train_gcn.py --num_layers 3 --lr 0.01 --epochs 100 --seed 42 --exp_name gcn_baseline

# 5-Seed Variance Evaluation (Seeds 42-46, 100 epochs) -> Table 3
for seed in 42 43 44 45 46; do
  python src/train_gcn.py --num_layers 3 --lr 0.01 --epochs 100 --seed $seed --exp_name gcn_seed_$seed
done

# Depth Sensitivity (L = 2, 3, 4, 5, 6; 80-epoch controlled budget) -> Table 1, Fig 2(a)
# Note: L = 3 at 80 epochs is the reference architecture checkpoint for controlled comparison
python src/train_gcn.py --num_layers 2 --lr 0.01 --epochs 80 --seed 42 --exp_name gcn_depth_2
python src/train_gcn.py --num_layers 3 --lr 0.01 --epochs 80 --seed 42 --exp_name gcn_depth_3
python src/train_gcn.py --num_layers 4 --lr 0.01 --epochs 80 --seed 42 --exp_name gcn_depth_4
python src/train_gcn.py --num_layers 5 --lr 0.01 --epochs 80 --seed 42 --exp_name gcn_depth_5
python src/train_gcn.py --num_layers 6 --lr 0.01 --epochs 80 --seed 42 --exp_name gcn_depth_6

# Learning Rate Sensitivity (eta = 0.001, 0.005, 0.01, 0.05; 80 epochs) -> Table 1, Fig 2(b)
# Note: eta = 0.01 at 80 epochs matches gcn_depth_3 as the baseline setting
python src/train_gcn.py --num_layers 3 --lr 0.001 --epochs 80 --seed 42 --exp_name gcn_lr_0.001
python src/train_gcn.py --num_layers 3 --lr 0.005 --epochs 80 --seed 42 --exp_name gcn_lr_0.005
python src/train_gcn.py --num_layers 3 --lr 0.010 --epochs 80 --seed 42 --exp_name gcn_lr_0.01
python src/train_gcn.py --num_layers 3 --lr 0.050 --epochs 80 --seed 42 --exp_name gcn_lr_0.05
```

### 3. Task 2: Mini-Batch GraphSAGE on `ogbn-products` (Table 2, Figures 3 & 4)
*All GraphSAGE runs default to seed 42 and 4 data loader workers (`--seed 42`).*
```bash
# Baseline GraphSAGE Reference (Fanout 15,10,5, Batch Size 1024, 15 epochs) -> Fig 3, Table 2
python src/train_sage.py --num_layers 3 --sizes 15,10,5 --batch_size 1024 --epochs 15 --seed 42 --exp_name sage_baseline

# Batch Size Sensitivity (B = 512, 1024, 2048; 5-epoch controlled budget) -> Table 2, Fig 4(a)
# Note: B = 1024 at 5 epochs serves as the ablation reference row (or epoch 5 of sage_baseline)
python src/train_sage.py --sizes 15,10,5 --batch_size 512 --epochs 5 --seed 42 --exp_name sage_batch_512
python src/train_sage.py --sizes 15,10,5 --batch_size 1024 --epochs 5 --seed 42 --exp_name sage_batch_1024
python src/train_sage.py --sizes 15,10,5 --batch_size 2048 --epochs 5 --seed 42 --exp_name sage_batch_2048

# Neighbor Sample Fanout Sensitivity (5-epoch budget) -> Table 2, Fig 4(b)
python src/train_sage.py --sizes 5,5,5 --batch_size 1024 --epochs 5 --seed 42 --exp_name sage_sampling_5_5_5
python src/train_sage.py --sizes 5,5,15 --batch_size 1024 --epochs 5 --seed 42 --exp_name sage_sampling_5_5_15
python src/train_sage.py --sizes 15,5,5 --batch_size 1024 --epochs 5 --seed 42 --exp_name sage_sampling_15_5_5
python src/train_sage.py --sizes 10,10,5 --batch_size 1024 --epochs 5 --seed 42 --exp_name sage_sampling_10_10_5
# Note: [15,10,5] at 5 epochs corresponds to sage_batch_1024 above.
```

### 4. Task 3: Full-Batch OOM Test & Comparative Profiling (Table 3, Figure 5)
```bash
# Full-Batch GCN on ogbn-products (OOM Test) -> results/oom_log.txt
# Note: train_gcn.py uses --dataset_dir (not --dataset)
python src/train_gcn.py --dataset_dir ./dataset --num_layers 3 \
  --hidden_channels 256 --epochs 1 --seed 42 \
  --exp_name gcn_oom_test 2>&1 | tee results/oom_log.txt

# Generate All Figures & Comparative Profiling Charts
python src/plot_results.py

# Verify all table values match the JSON logs (exits 0 on success, 1 on mismatch)
python src/validate_tables.py
```
This generates:
- `gcn_convergence.png`: Training/Validation/Test loss & accuracy over 100 epochs (Fig 1).
- `gcn_depth_ablation.png`: Accuracy vs number of GCN layers (Fig 2a).
- `gcn_lr_ablation.png`: Accuracy vs learning rate on log scale (Fig 2b).
- `sage_convergence.png`: GraphSAGE loss and train/val/test accuracy curves (Fig 3).
- `sage_batch_ablation.png`: GraphSAGE accuracy vs mini-batch size (Fig 4a).
- `sage_sampling_ablation.png`: GraphSAGE accuracy vs neighbor sampling fanouts (Fig 4b).
- `comparative_profiling.png`: Peak GPU memory & epoch runtime (log scale) cross-benchmark comparison (Fig 5).
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

### 2. Task 1: Full-Batch GCN on `ogbn-arxiv`
```bash
# Baseline GCN (Depth 3, LR 0.01)
python src/train_gcn.py --num_layers 3 --lr 0.01 --epochs 100 --exp_name gcn_baseline

# Depth Analysis (e.g. 2, 4, 5, 6)
python src/train_gcn.py --num_layers 2 --exp_name gcn_depth_2
python src/train_gcn.py --num_layers 4 --exp_name gcn_depth_4

# Learning Rate Analysis (e.g. 0.001, 0.005, 0.05)
python src/train_gcn.py --lr 0.001 --exp_name gcn_lr_0.001
python src/train_gcn.py --lr 0.05 --exp_name gcn_lr_0.05
```

### 3. Task 2: Mini-Batch GraphSAGE on `ogbn-products`
```bash
# Baseline GraphSAGE (Fanout 15,10,5, Batch Size 1024)
python src/train_sage.py --num_layers 3 --sizes 15,10,5 --batch_size 1024 --epochs 15 --exp_name sage_baseline

# Batch Size Sensitivity (e.g. 512, 2048, 4096)
python src/train_sage.py --batch_size 512 --exp_name sage_batch_512
python src/train_sage.py --batch_size 2048 --exp_name sage_batch_2048

# Neighbor Sample Size Sensitivity (e.g. 5,5,5 vs 10,10,5)
python src/train_sage.py --sizes 5,5,5 --exp_name sage_sampling_5_5_5
python src/train_sage.py --sizes 10,10,5 --exp_name sage_sampling_10_10_5
```

### 4. Task 3: Generate Figures & Comparative Profiling
```bash
python src/plot_results.py
```
This generates:
- `gcn_convergence.png`: Training/Validation/Test loss & accuracy over epochs.
- `gcn_depth_ablation.png`: Accuracy vs number of GCN layers.
- `gcn_lr_ablation.png`: Accuracy vs learning rate.
- `sage_convergence.png`: GraphSAGE loss and accuracy curves.
- `sage_batch_ablation.png`: Accuracy vs batch size.
- `comparative_profiling.png`: Peak memory consumption & average runtime per epoch comparison.
# CSC 591/791 Homework 2: GCN vs. GraphSAGE & Scalability Analysis
**Course**: Deep Graph Learning  
**Date**: October 2026  
**Submission**: Code, Empirical Results, and Technical Report  

---

## Abstract
Graph Convolutional Networks (GCN) and GraphSAGE represent foundational paradigms in geometric deep learning. While full-batch GCN operates across the entire graph topology, it suffers from the "neighborhood explosion" bottleneck on large graphs. GraphSAGE overcomes this via stochastic neighborhood sampling and mini-batch optimization. In this report, we evaluate both architectures on benchmark datasets: `ogbn-arxiv` and `ogbn-products`. We systematically analyze model depth, learning rates, batch sizes, and sampling budgets, presenting empirical profiling of memory scalability and runtime efficiency.

---

## 1. Task 1: Full-Batch GCN Reproduction & Convergence (`ogbn-arxiv`)

### 1.1 Experimental Configuration
Full-batch GCN was trained on the `ogbn-arxiv` citation network ($|V| = 169,343$, $|E| = 1,166,243$, 40 subject classes). The baseline architecture employs a 3-layer GCN with hidden dimension $d = 256$, batch normalization, dropout $p = 0.5$, and ReLU activations, optimized using Adam ($\eta = 0.01$).

### 1.2 Convergence Dynamics
GCN exhibits rapid, monotonic convergence during the initial 40 epochs, reducing training cross-entropy loss from $4.21$ to $1.05$. Validation accuracy plateaus around epoch 70, reaching a peak validation accuracy of **71.82%** and a corresponding test accuracy of **71.30%** (Figure 1). This faithfully replicates the official Open Graph Benchmark (OGB) leaderboard baseline ($71.74\% \pm 0.29\%$).

![GCN Convergence](../results/plots/gcn_convergence.png)
*Figure 1: Training, validation, and test loss and accuracy curves for 3-layer GCN on `ogbn-arxiv` over 100 epochs.*

---

### 1.3 Hyperparameter Sensitivity Analysis

| Configuration | Best Val Acc (%) | Best Test Acc (%) | Peak Memory (MB) | Avg Epoch Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Depth $L = 2$ | 70.55% | 69.66% | 5,445.7 MB | 0.0100 s |
| **Depth $L = 3$ (Base)** | **71.82%** | **71.30%** | **6,030.5 MB** | **0.0145 s** |
| Depth $L = 4$ | 71.60% | 70.65% | 6,616.2 MB | 0.0156 s |
| Depth $L = 5$ | 71.04% | 70.08% | 7,201.7 MB | 0.0189 s |
| Depth $L = 6$ | 70.32% | 69.72% | 7,788.0 MB | 0.0223 s |
| LR $\eta = 0.001$ | 68.91% | 67.71% | 6,030.5 MB | 0.0146 s |
| LR $\eta = 0.005$ | 71.63% | 70.87% | 6,030.5 MB | 0.0135 s |
| **LR $\eta = 0.010$ (Base)** | **71.82%** | **71.30%** | **6,030.5 MB** | **0.0145 s** |
| LR $\eta = 0.050$ | 69.33% | 68.17% | 6,030.5 MB | 0.0132 s |

<p align="center">
  <img src="../results/plots/gcn_depth_ablation.png" width="48%" />
  <img src="../results/plots/gcn_lr_ablation.png" width="48%" />
</p>
*Figure 2: GCN accuracy vs. model depth (left) and learning rate (right).*

#### Discussion:
- **Impact of Model Depth ($L$)**: Increasing depth from $L=2$ to $L=3$ expands the receptive field, improving test accuracy from $69.66\%$ to $71.30\%$. However, increasing depth further ($L=4, 5, 6$) causes monotonic performance degradation down to $69.72\%$. This deterioration is attributed to the **oversmoothing problem**: recursive normalized Laplacian smoothing acts as a low-pass filter on graph signals. As $L \to \infty$, node representations converge to a stationary distribution proportional to node degree, collapsing feature distinguishability across classes. Peak memory also grows linearly from 5,445 MB ($L=2$) to 7,788 MB ($L=6$) due to storing full activation maps across all layers.
- **Impact of Learning Rate ($\eta$)**: A small learning rate ($\eta = 0.001$) converges sluggishly within 80 epochs, underfitting at $67.71\%$ test accuracy. Conversely, an overly large learning rate ($\eta = 0.05$) induces gradient oscillations and settles into sub-optimal plateaus ($68.17\%$). Optimal performance is achieved at $\eta = 0.01$ ($71.30\%$).

---

## 2. Task 2: Mini-Batch GraphSAGE & Neighborhood Sampling (`ogbn-products`)

### 2.1 Experimental Configuration
To evaluate scalability, mini-batch GraphSAGE was evaluated on the large-scale `ogbn-products` graph ($|V| = 2,449,029$, $|E| = 61,859,140$, 47 classes). The architecture utilizes 3 layers ($d = 256$), mean aggregation, dropout $p = 0.5$, and batch normalization.

### 2.2 Convergence & Replication
GraphSAGE exhibited steady convergence across 15 epochs (Figure 3), achieving a peak validation accuracy of **91.61%** and a test accuracy of **78.61%**, closely aligning with the official OGB GraphSAGE benchmark ($78.70\% \pm 0.36\%$).

![GraphSAGE Convergence](../results/plots/sage_convergence.png)
*Figure 3: GraphSAGE loss and accuracy over 15 epochs on `ogbn-products` ($B=1024, S=[15,10,5]$).*

---

### 2.3 Sampling Hyperparameter Investigation

| Configuration | Best Val Acc (%) | Best Test Acc (%) | Peak Memory (MB) | Avg Epoch Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Batch $B = 512$ | 90.80% | 76.45% | 7,113.8 MB | 66.29 s |
| **Batch $B = 1024$ (Base)** | **91.61%** | **78.61%** | **7,109.7 MB** | **60.64 s** |
| Batch $B = 2048$ | 90.62% | 77.33% | 9,393.7 MB | 49.94 s |
| Fanout $S=[5, 5, 5]$ | 89.05% | 75.92% | **3,735.0 MB** | **16.92 s** |
| Fanout $S=[10, 10, 5]$ | 90.37% | 77.24% | 6,152.9 MB | 46.77 s |
| **Fanout $S=[15, 10, 5]$** | **91.61%** | **78.61%** | **7,109.7 MB** | **60.64 s** |

<p align="center">
  <img src="../results/plots/sage_batch_ablation.png" width="48%" />
  <img src="../results/plots/sage_sampling_ablation.png" width="48%" />
</p>
*Figure 4: GraphSAGE sensitivity to mini-batch size (left) and layer-wise neighbor sampling fanout (right).*

#### Discussion:
- **Impact of Batch Size ($B$)**: Increasing batch size from 512 to 2048 reduces iteration overhead and improves GPU vector parallelization, dropping per-epoch runtime from $66.29$s to $49.94$s. However, larger batch sizes expand the multi-hop union frontier, driving peak memory up to $9,393.7$ MB. Batch size $B = 1024$ delivers the optimal trade-off between statistical gradient quality and memory ($78.61\%$ test accuracy).
- **Impact of Neighbor Sample Sizes ($S$)**: Varying the sampling budget reveals a dramatic efficiency-accuracy trade-off. Pruning the fanout from $[15, 10, 5]$ to $[5, 5, 5]$ slashes epoch training time by **3.6$\times$** ($60.64$s $\to 16.92$s) and cuts peak memory nearly in half ($7,109.7$ MB $\to 3,735.0$ MB), at a modest accuracy cost ($75.92\%$ vs. $78.61\%$). This confirms that smaller sampling budgets are exceptionally effective for fast prototyping and memory-constrained deployments.

---

## 3. Task 3: Comparative Profiling & Scalability Discussion

### 3.1 Empirical Profiling Comparison

| Metric | Full-Batch GCN | Mini-Batch GraphSAGE |
| :--- | :---: | :---: |
| **Dataset** | `ogbn-arxiv` | `ogbn-products` |
| **Number of Nodes ($|V|$)** | 169,343 | 2,449,029 ($14.5\times$) |
| **Number of Edges ($|E|$)** | 1,166,243 | 61,859,140 ($53.0\times$) |
| **Training Paradigm** | Full-graph dense/CSR | Stochastic mini-batch |
| **Peak GPU Memory** | 6,030.5 MB | 7,109.7 MB (or 3,735.0 MB) |
| **Avg Epoch Runtime** | 0.0145 s | 60.64 s (or 16.92 s) |
| **Test Accuracy** | 71.30% | 78.61% |

![Comparative Profiling](../results/plots/comparative_profiling.png)
*Figure 5: Comparative peak memory consumption and epoch runtime between Full-Batch GCN and Mini-Batch GraphSAGE.*

Full-batch GCN computes an entire epoch in only $14.5$ ms on `ogbn-arxiv` but requires $6.0$ GB of GPU memory for a graph with only 169K nodes. In contrast, `ogbn-products` contains **2.45 million nodes and 61.8 million edges** ($53\times$ more edges). While full-batch GCN would catastrophically crash with an Out-of-Memory (OOM) error on any standard GPU, mini-batch GraphSAGE trains stably using only $7.1$ GB VRAM (and only $3.7$ GB under fanout $[5,5,5]$).

---

### 3.2 Theoretical Scalability Analysis: The Neighborhood Explosion Bottleneck

The scalability dichotomy between GCN and GraphSAGE stems from the receptive field structure during backpropagation:

#### Full-Batch GCN Explosion:
In full-batch GCN, computing node embeddings at layer $l+1$ requires the global graph adjacency matrix:
$$H^{(l+1)} = \sigma\left(\tilde{A}_{\text{norm}} H^{(l)} W^{(l)}\right)$$
To compute exact gradients $\frac{\partial \mathcal{L}}{\partial W^{(l)}}$, automatic differentiation must retain the full activation matrices $\{H^{(0)}, H^{(1)}, \dots, H^{(L)}\}$ in GPU memory simultaneously, requiring $\mathcal{O}(L \cdot |V| \cdot d)$ space, alongside edge storage $\mathcal{O}(|E|)$. For web-scale graphs where $|V| \ge 10^7$ and $|E| \ge 10^9$, full-batch memory overhead becomes impossible.

Furthermore, naive node-wise mini-batching without sampling fails due to **exponential receptive field expansion**. In real-world graphs exhibiting power-law degree distributions with dense hubs, the size of a node's receptive field across $L$ layers expands as $\mathcal{O}(d_{\text{avg}}^L)$. For a 3-layer model on a graph with average degree $d_{\text{avg}} = 25$, each batch node's receptive field touches $\sim 25^3 \approx 15,625$ nodes; across a batch of $B = 1024$, the union of multi-hop neighborhoods collapses into the entire graph ($|V|$), causing exponential memory blow-up.

#### Stochastic Sampling Solution (GraphSAGE):
GraphSAGE circumvents this bottleneck by restricting aggregation to a fixed uniform sample of size $S_l$ at each hop $l \in \{1, \dots, L\}$. For a mini-batch of $B$ target nodes, the maximum computational graph size is strictly bounded by:
$$\text{Sampled Nodes} \le B \cdot \prod_{l=1}^L S_l$$
This caps per-batch memory to $\mathcal{O}\left(B \cdot d \cdot \prod_{l=1}^L S_l\right)$, which is **strictly independent of total graph size $|V|$ and $|E|$**. Consequently, stochastic neighborhood sampling decouples training memory from graph scale, enabling scalable representation learning on graphs of arbitrary magnitude.

---

## 4. Conclusion
This study reproduced node classification for GCN and GraphSAGE on OGB benchmarks, verifying that depth $L=3$ and learning rate $\eta=0.01$ optimize GCN, while batch size $B=1024$ and fanout $[15,10,5]$ optimize GraphSAGE. Profiling confirmed that stochastic neighborhood sampling effectively prevents the neighborhood explosion, providing bounded memory scalability on massive graph structures.

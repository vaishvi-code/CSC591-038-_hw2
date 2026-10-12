# CSC 591/791 Homework 2: GCN vs. GraphSAGE & Scalability Analysis
**Author**: Vaishvi Patel  
**Course**: CSC 591/791 Deep Graph Learning, North Carolina State University  
**Repository**: [https://github.com/vaishvi-code/CSC591-038-_hw2](https://github.com/vaishvi-code/CSC591-038-_hw2)  
**Date**: October 2026  

---

## Abstract
Graph Convolutional Networks (GCN) and GraphSAGE represent foundational paradigms in geometric deep learning. While full-batch GCN operates across the entire graph topology, it encounters steep memory scaling bottlenecks on large graphs. GraphSAGE addresses this via stochastic neighborhood sampling and mini-batch optimization. In this report, we evaluate both architectures across benchmark datasets: `ogbn-arxiv` and `ogbn-products`. We examine model depth, learning rates, mini-batch sizes, and neighbor sampling budgets under controlled training lengths, providing empirical profiling of runtime efficiency and GPU memory allocation.

---

## 1. Task 1: Full-Batch GCN Reproduction & Convergence (`ogbn-arxiv`)

### 1.1 Experimental Configuration
Full-batch GCN was evaluated on the `ogbn-arxiv` citation network ($|V| = 169,343$, $|E| = 1,166,243$, 40 subject classes). The baseline architecture employs a 3-layer GCN with hidden dimension $d = 256$, batch normalization, dropout $p = 0.5$, and ReLU activations, trained with Adam and a constant learning rate ($\eta = 0.01$). The baseline converged over 100 epochs, while all hyperparameter sensitivity runs were executed under a standardized 80-epoch budget.

### 1.2 Convergence Dynamics
GCN exhibits rapid loss reduction during the initial 40 epochs (training loss declining from $4.21$ to $1.05$). Following transient validation fluctuations in early epochs, validation accuracy stabilizes around epoch 70. At 80 epochs, the baseline achieves $71.49\%$ validation accuracy and $70.88\%$ test accuracy; extending training to 100 epochs yields a peak validation accuracy of **71.82%** and a test accuracy of **71.30%** (Figure 1). This is within $\sim 0.4$ percentage points (pp) of the official Open Graph Benchmark (OGB) leaderboard baseline ($71.74\% \pm 0.29\%$), despite training for 100 epochs versus the 500 epochs used in the reference submission.

![GCN Convergence](../results/plots/gcn_convergence.png)
*Figure 1: Training, validation, and test loss and accuracy curves for 3-layer GCN on `ogbn-arxiv` over 100 epochs.*

---

### 1.3 Hyperparameter Sensitivity Analysis

| Configuration | Best Val Acc (%) | Best Test Acc (%) | Peak Memory (MB) | Avg Epoch Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Depth $L = 2$ | 70.55% | 69.66% | 5,445.7 MB | 0.0100 s |
| **Depth $L = 3$ (80 ep)** | **71.49%** | **70.88%** | **6,030.5 MB** | **0.0145 s** |
| *Depth $L = 3$ (100 ep base)* | *71.82%* | *71.30%* | *6,030.5 MB* | *0.0145 s* |
| Depth $L = 4$ | 71.60% | 70.65% | 6,616.2 MB | 0.0156 s |
| Depth $L = 5$ | 71.04% | 70.08% | 7,201.7 MB | 0.0189 s |
| Depth $L = 6$ | 70.32% | 69.72% | 7,788.0 MB | 0.0223 s |
| LR $\eta = 0.001$ | 68.91% | 67.71% | 6,030.5 MB | 0.0146 s |
| LR $\eta = 0.005$ | 71.63% | 70.87% | 6,030.5 MB | 0.0135 s |
| **LR $\eta = 0.010$ (Base)** | **71.49%** | **70.88%** | **6,030.5 MB** | **0.0145 s** |
| LR $\eta = 0.050$ | 69.33% | 68.17% | 6,030.5 MB | 0.0132 s |

<p align="center">
  <img src="../results/plots/gcn_depth_ablation.png" width="48%" />
  <img src="../results/plots/gcn_lr_ablation.png" width="48%" />
</p>
*Figure 2: GCN accuracy across model depths (left) and learning rates (right).*

#### Discussion:
- **Impact of Model Depth ($L$)**: Increasing depth from $L=2$ to $L=3$ expands the receptive field to 3 hops, improving test accuracy from $69.66\%$ to $70.88\%$. Beyond $L=3$, test accuracy declines monotonically from $70.88\%$ ($L=3$) to $69.72\%$ ($L=6$), representing a $1.16$ pp drop. Because batch normalization is employed, this degradation is attributed to a combination of optimization difficulty (e.g., gradient attenuation during backpropagation) alongside early oversmoothing tendencies, where recursive normalized graph convolutions progressively smooth feature variance toward the stationary eigenvector proportional to $\sqrt{\tilde{d}_i}$. Peak allocated memory (`torch.cuda.max_memory_allocated`) grows linearly by $\approx 585$ MB per additional layer (from 5,445 MB at $L=2$ to 7,788 MB at $L=6$) due to caching intermediate activation maps for backpropagation.
- **Impact of Learning Rate ($\eta$)**: A small learning rate ($\eta = 0.001$) converges slowly and underfits within 80 epochs ($67.71\%$). An aggressive learning rate ($\eta = 0.05$) exhibits higher gradient variance and plateaus at sub-optimal accuracy ($68.17\%$). Within the tested grid, $\eta = 0.01$ achieves the strongest convergence ($70.88\%$). Small runtime variations across LR settings ($0.013$--$0.015$s) reflect host scheduling noise.

---

## 2. Task 2: Mini-Batch GraphSAGE & Neighborhood Sampling (`ogbn-products`)

### 2.1 Experimental Configuration
GraphSAGE was evaluated on the large-scale `ogbn-products` graph ($|V| = 2,449,029$, $|E| = 61,859,140$, 47 classes). The baseline utilizes 3 layers ($d = 256$), mean aggregation, dropout $p = 0.5$, and batch normalization.

### 2.2 Convergence & Replication
GraphSAGE converged steadily over 15 epochs (Figure 3), achieving **91.61%** validation accuracy and **78.61%** test accuracy, matching the official OGB benchmark ($78.70\% \pm 0.36\%$). The notable validation-test gap ($91.61\%$ vs. $78.61\%$) is expected: OGB products uses a sales-ranking-based split where popular products comprise training and tail products form test sets, introducing an intentional distribution shift across splits.

![GraphSAGE Convergence](../results/plots/sage_convergence.png)
*Figure 3: GraphSAGE training loss and validation/test accuracy over 15 epochs on `ogbn-products` ($B=1024, S=[15,10,5]$).*

---

### 2.3 Sampling Hyperparameter Investigation

| Configuration | Best Val Acc (%) | Best Test Acc (%) | Peak Memory (MB) | Avg Epoch Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Batch $B = 512$ | 90.80% | 76.45% | 7,113.8 MB | 66.29 s |
| **Batch $B = 1024$ (5 ep)** | **90.77%** | **77.89%** | **7,109.7 MB** | **59.45 s** |
| *Batch $B = 1024$ (15 ep base)* | *91.61%* | *78.61%* | *7,109.7 MB* | *60.64 s* |
| Batch $B = 2048$ | 90.62% | 77.33% | 9,393.7 MB | 49.94 s |
| Fanout $S=[5, 5, 5]$ | 89.05% | 75.92% | **3,735.0 MB** | **16.92 s** |
| Fanout $S=[10, 10, 5]$ | 90.37% | 77.24% | 6,152.9 MB | 46.77 s |
| **Fanout $S=[15, 10, 5]$ (5 ep)** | **90.77%** | **77.89%** | **7,109.7 MB** | **59.45 s** |

<p align="center">
  <img src="../results/plots/sage_batch_ablation.png" width="48%" />
  <img src="../results/plots/sage_sampling_ablation.png" width="48%" />
</p>
*Figure 4: GraphSAGE sensitivity to mini-batch size (left) and neighbor fanout (right).*

#### Discussion:
- **Impact of Batch Size ($B$)**: Increasing batch size from 512 to 2048 decreases epoch runtime from $66.29$s to $49.94$s. With 196,615 training nodes, larger batches reduce total iteration steps per epoch ($\approx 385$ steps at $B=512$, $\approx 193$ at $B=1024$, and $\approx 97$ at $B=2048$), amortizing loop overhead and GPU kernel launch latency. Peak allocated memory remains nearly identical between $B=512$ and $B=1024$ ($\approx 7,110$ MB) because the peak memory ceiling is dominated by the full test evaluation pass ($B_{\text{eval}} = 4096$). At $B=2048$, the larger training multi-hop union frontier elevates peak training memory to $9,393.7$ MB. Within the tested 5-epoch budget, $B=1024$ achieves the strongest test accuracy ($77.89\%$).
- **Impact of Sampling Fanout ($S$)**: Comparing fanout configurations at 5 epochs reveals a clear trade-off: reducing fanout from $[15, 10, 5]$ to $[5, 5, 5]$ provides a **3.5$\times$ speedup** ($59.45$s $\to 16.92$s) and reduces peak allocated memory by $47\%$ ($7,109.7$ MB $\to 3,735.0$ MB), at the cost of a $1.97$ pp drop in test accuracy ($77.89\% \to 75.92\%$). Note that measured memory scales by $1.9\times$ rather than the theoretical $6\times$ sampling ratio because static node features ($2.45\text{M} \times 100 \times 4\text{ bytes} \approx 980$ MB), graph indexing pointers, and PyTorch allocator caching establish a constant memory baseline.

---

## 3. Task 3: Comparative Profiling & Scalability Discussion

### 3.1 Empirical Profiling Comparison

| Metric | Full-Batch GCN | Mini-Batch GraphSAGE |
| :--- | :---: | :---: |
| **Dataset** | `ogbn-arxiv` | `ogbn-products` |
| **Graph Scale ($|V|, |E|$)** | 169K nodes, 1.16M edges | **2.45M nodes, 61.8M edges** |
| **Training Steps per Epoch** | 1 step (full graph) | $\approx 193$ steps (mini-batch) |
| **Time per Step** | $\approx 14.5$ ms | $\approx 310$ ms |
| **Time per Epoch** | $0.0145$ s | $60.64$ s ($16.92$ s for $[5,5,5]$) |
| **Measured Peak GPU Mem** | 6,030.5 MB | 7,109.7 MB (3,735.0 MB for $[5,5,5]$) |
| **Slope Extrapolation on Products** | $\approx 25.5$ GB (Activations) | N/A (Fits on 15 GB GPU) |
| **Empirical Test on Products** | Out-Of-Memory (OOM) | Runs stably |
| **Test Accuracy** | 71.30% (100 ep) | 78.61% (15 ep) |

![Comparative Profiling](../results/plots/comparative_profiling.png)
*Figure 5: Comparative peak GPU memory and epoch runtime across paradigms on their respective benchmarks.*

Because the two paradigms operate on different benchmarks, the runtime per epoch reflects both algorithm and dataset scale: full-batch GCN completes an epoch in a single $14.5$ ms gradient step on `ogbn-arxiv`, while mini-batch GraphSAGE performs $\approx 193$ sequential sampled steps per epoch on `ogbn-products`.

---

### 3.2 Hardware Scaling & Slope-Based Extrapolation
On `ogbn-arxiv`, Table 1 demonstrates that full-batch GCN activation memory increases by $\approx 585$ MB per layer for 169K nodes. When scaling to `ogbn-products` ($2.45\text{M}$ nodes, a $14.5\times$ increase), the activation memory slope scales proportionally:
$$\Delta M_{\text{layer}} \approx 585\text{ MB} \times 14.5 \approx 8.5\text{ GB per layer}$$
For a 3-layer model, caching activations alone requires an estimated $\approx 25.5\text{ GB}$. Adding edge indices for 61.8 million edges ($\approx 1\text{ GB}$) and intermediate aggregation buffers pushes the memory requirement well beyond 30 GB. When executing full-batch GCN on `ogbn-products` on an NVIDIA Tesla T4 GPU (15.0 GB VRAM), PyTorch immediately halts with a `torch.cuda.OutOfMemoryError`. In contrast, mini-batch GraphSAGE trains stably on the same GPU, consuming $7.1$ GB VRAM (and only $3.7$ GB under $[5,5,5]$).

---

### 3.3 Theoretical Scalability Analysis: Resolving Neighborhood Explosion
The memory divergence arises from the receptive field expansion during backpropagation:

#### Full-Batch Activation Bottleneck:
Full-batch GCN computes layer activations globally: $H^{(l+1)} = \sigma(\tilde{A}_{\text{norm}} H^{(l)} W^{(l)})$. Exact gradient computation requires retaining all intermediate activation tensors $\{H^{(0)}, \dots, H^{(L)}\}$, imposing an $\mathcal{O}(L \cdot |V| \cdot d)$ memory footprint.

Furthermore, naive node-wise mini-batching without sampling fails due to exponential receptive field growth. In real-world graphs exhibiting power-law degree distributions ($d_{\text{avg}} = |E| / |V| \approx 25.3$ directed, $\approx 50.5$ undirected on products), an unconstrained $L$-hop receptive field expands with degree branching. For $L=3$, the theoretical neighborhood paths expand as $(50.5)^3 \approx 1.28 \times 10^5$; across a mini-batch of $B = 1024$ seed nodes, the union of receptive fields rapidly encompasses nearly the entire graph.

#### Stochastic Neighborhood Sampling:
GraphSAGE restricts neighbor aggregation to a fixed sample budget $S_l$ at each hop $l \in \{1, \dots, L\}$. For a mini-batch of $B$ seed nodes, the computational subgraph is strictly bounded by:
$$\text{Sampled Seed Frontier} \le B \cdot \prod_{l=1}^L S_l$$
This bounds per-step backpropagation activation memory to $\mathcal{O}(B \cdot d \cdot \prod_{l=1}^L S_l)$. While static node features and graph structure still occupy $\mathcal{O}(|V| \cdot d + |E|)$ host memory, gradient computation memory is decoupled from total graph size $|V|$, enabling scalable representation learning on graphs of arbitrary magnitude.

---

## 4. Reproducibility & Environment Details
All results are reproducible up to CUDA hardware nondeterminism using the scripts, random seed 42, and logs provided in the GitHub repository ([https://github.com/vaishvi-code/CSC591-038-_hw2](https://github.com/vaishvi-code/CSC591-038-_hw2)):
- **Hardware**: NVIDIA Tesla T4 GPU (15.0 GB VRAM), Intel Xeon CPU @ 2.20GHz (2 vCPUs), 12.7 GB System RAM.
- **Software**: Python 3.13, PyTorch 2.11.0+cu130, PyTorch Geometric 2.4.0, OGB 1.3.6.
- **Optimization**: Adam optimizer, constant learning rate as specified, single seed 42.

---

## 5. Conclusion
This study reproduced GCN and GraphSAGE on OGB benchmarks, observing that $L=3, \eta=0.01$ and $B=1024, S=[15,10,5]$ provided the most effective configurations within the explored parameter grid under controlled training lengths. Profiling confirmed that stochastic neighborhood sampling prevents receptive field explosion during backpropagation, enabling tractable mini-batch training on large-scale graphs where full-batch GCN encounters out-of-memory constraints.

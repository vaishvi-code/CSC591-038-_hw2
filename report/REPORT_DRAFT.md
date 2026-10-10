# CSC 591/791 Homework 2: GCN vs. GraphSAGE & Scalability Analysis
## Written Report Outline & Key Talking Points (Target: ~3 Pages)

---

### Section 1: Full-Batch GCN Reproduction & Convergence (ogbn-arxiv)
- **Experimental Setup**:
  - Dataset: `ogbn-arxiv` (169,343 nodes, 1,166,243 edges, 40 classes).
  - Baseline Architecture: 3-layer GCN with hidden dimension 256, Batch Normalization, Dropout ($p=0.5$), and ReLU. Optimizer: Adam with $\text{lr}=0.01$.
- **Convergence Behavior**:
  - Insert Figure: `results/plots/gcn_convergence.png`
  - Observation: GCN steadily reduces training cross-entropy loss and reaches ~71-73% test accuracy on `ogbn-arxiv`.
  - Discuss difference between validation and test curve trends.
- **Hyperparameter Sensitivity**:
  - **Model Depth ($L \in \{2, 3, 4, 5, 6\}$)**:
    - Insert Figure: `results/plots/gcn_depth_ablation.png`
    - Discussion: Peak accuracy occurs around $L=3$. As depth increases ($L \ge 5$), accuracy deteriorates due to the **oversmoothing problem** (node representations become indistinguishable as repeated Laplacian smoothing averages node features across distant components) and vanishing gradients.
  - **Learning Rate ($\text{lr} \in \{0.001, 0.005, 0.01, 0.05\}$)**:
    - Insert Figure: `results/plots/gcn_lr_ablation.png`
    - Discussion: A very small learning rate ($0.001$) converges sluggishly; an overly large learning rate ($0.05$) induces instability/oscillation or sub-optimal plateaus. Optimal convergence is achieved near $0.01$.

---

### Section 2: Mini-Batch GraphSAGE & Neighborhood Sampling (ogbn-products)
- **Experimental Setup**:
  - Dataset: `ogbn-products` (2,449,029 nodes, 61,859,140 edges, 47 classes).
  - Architecture: GraphSAGE (Mean Aggregator), 3 layers, hidden dimension 256. Mini-batching via `NeighborSampler`.
- **Convergence Behavior**:
  - Insert Figure: `results/plots/sage_convergence.png`
  - Observation: Mini-batch stochastic training achieves rapid convergence with mini-batch updates without needing the full adjacency matrix in GPU memory. Test accuracy reaches ~78-79%.
- **Hyperparameter Sensitivity**:
  - **Layer-wise Neighbor Sample Sizes (Fanout)**:
    - Compare configurations (e.g., `[5, 5, 5]` vs `[10, 10, 5]` vs `[15, 10, 5]`).
    - Discussion: Smaller fanout sizes reduce per-batch computation and memory footprint, but introduce higher variance in neighborhood aggregation. Larger fanouts improve representation stability and accuracy at the cost of higher sampling overhead and increased batch subgraph size.
  - **Batch Size ($B \in \{512, 1024, 2048, 4096\}$)**:
    - Insert Figure: `results/plots/sage_batch_ablation.png`
    - Discussion: Small batch sizes ($512$) lead to noisy gradient updates and lower GPU tensor utilization. Excessively large batches ($4096$) suffer from larger frontier trees and increased memory overhead. A balanced batch size (e.g., $1024$) offers optimal throughput and steady convergence.

---

### Section 3: Comparative Profiling & Scalability Analysis
- **Comparative Profiling**:
  - Insert Figure: `results/plots/comparative_profiling.png`
  - Table comparing Full-Batch GCN vs Mini-Batch GraphSAGE:
    | Metric | Full-Batch GCN (`ogbn-arxiv`) | Mini-Batch GraphSAGE (`ogbn-products`) |
    | :--- | :--- | :--- |
    | **Graph Size (|V|, |E|)** | 169K nodes, 1.16M edges | 2.45M nodes, 61.8M edges |
    | **Peak Memory (MB)** | [Insert from profiling] | [Insert from profiling] |
    | **Avg Epoch Time (s)** | [Insert from profiling] | [Insert from profiling] |
    | **Training Paradigm** | Full Adjacency in Memory | Fixed Fanout Subgraphs |
- **Theoretical Explanation: Preventing Neighborhood Explosion**:
  - In full-batch GCN, computing layer $l+1$ node states requires full graph neighborhood propagation: $H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$. For web-scale graphs (like `ogbn-products` with 2.4M nodes and 61.8M edges), storing the entire feature matrix ($2.45\text{M} \times 100 \times 4\text{ bytes} \approx 1\text{ GB}$) and intermediate activation matrices across all layers simultaneously in GPU VRAM leads to out-of-memory (OOM) errors.
  - Furthermore, if a naive mini-batching method is used without sampling, a node's receptive field expands exponentially with depth $L$: in scale-free power-law graphs with high-degree hubs, the receptive field quickly encompasses nearly the entire graph ($O(d_{\text{avg}}^L)$ nodes).
  - **GraphSAGE Solution**: Stochastic neighborhood sampling caps the maximum neighbors sampled per node at layer $l$ to a fixed budget $S_l$. A mini-batch of size $B$ across $L$ layers is bounded strictly to $O(B \cdot \prod_{l=1}^L S_l)$ nodes, decoupling memory footprint from total graph size $|V|$ and enabling training on arbitrary-scale graphs.

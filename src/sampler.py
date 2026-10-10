import torch

class FastTorchNeighborSampler:
    """
    Pure PyTorch Vectorized Neighbor Sampler for GraphSAGE.
    Zero external C++ dependencies (no pyg-lib or torch-sparse required).
    Works universally on CPU and GPU across any PyTorch version.
    """
    def __init__(self, edge_index, num_nodes, device='cpu'):
        self.num_nodes = num_nodes
        self.device = torch.device(device)

        row = edge_index[0].to(self.device)
        col = edge_index[1].to(self.device)

        perm = torch.argsort(row)
        self.col = col[perm]
        deg = torch.bincount(row[perm], minlength=num_nodes)
        self.indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=self.device)
        self.indptr[1:] = torch.cumsum(deg, dim=0)

        # Lookup table for O(1) vectorized node ID remapping
        self.lookup = torch.full((num_nodes,), -1, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def sample(self, seed_nodes, sample_sizes):
        """
        Samples a multi-hop neighborhood for seed_nodes.
        Returns:
            ordered_nodes: 1D tensor where indices [0 : len(seed_nodes)] are seed_nodes
            sub_edge_index: [2, E_sub] edges remapped to indices in ordered_nodes
        """
        seed_nodes = seed_nodes.to(self.device)
        current_nodes = seed_nodes
        all_edges_src = []
        all_edges_dst = []
        all_nodes = [seed_nodes]

        for s in sample_sizes:
            starts = self.indptr[current_nodes]
            ends = self.indptr[current_nodes + 1]
            degs = ends - starts
            valid = degs > 0

            if not valid.any():
                break

            valid_nodes = current_nodes[valid]
            valid_starts = starts[valid]
            valid_degs = degs[valid]

            offsets = (torch.rand((len(valid_nodes), s), device=self.device) * valid_degs.unsqueeze(1)).long()
            idx = valid_starts.unsqueeze(1) + offsets
            nbrs = self.col[idx]

            src = nbrs.view(-1)
            dst = valid_nodes.unsqueeze(1).expand(-1, s).contiguous().view(-1)

            all_edges_src.append(src)
            all_edges_dst.append(dst)

            unique_nbrs = torch.unique(src)
            all_nodes.append(unique_nbrs)
            current_nodes = unique_nbrs

        all_nodes_flat = torch.cat(all_nodes)
        unique_nodes = torch.unique(all_nodes_flat)

        # Order nodes so that seed_nodes are strictly at indices [0 : len(seed_nodes)]
        self.lookup[seed_nodes] = 1
        is_seed = (self.lookup[unique_nodes] == 1)
        self.lookup[seed_nodes] = -1
        other_nodes = unique_nodes[~is_seed]

        ordered_nodes = torch.cat([seed_nodes, other_nodes])

        # Remap global node IDs to local indices [0 : len(ordered_nodes)-1]
        self.lookup[ordered_nodes] = torch.arange(len(ordered_nodes), dtype=torch.long, device=self.device)

        if all_edges_src:
            edges_src = torch.cat(all_edges_src)
            edges_dst = torch.cat(all_edges_dst)
            local_src = self.lookup[edges_src]
            local_dst = self.lookup[edges_dst]
            sub_edge_index = torch.stack([local_src, local_dst], dim=0)
        else:
            sub_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        # Clean up lookup table
        self.lookup[ordered_nodes] = -1

        return ordered_nodes, sub_edge_index

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.cluster import KMeans

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CausalTransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        B, T, C = x.shape
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        if not hasattr(self, 'causal_mask') or self.causal_mask.size(0) != T:
            mask = self._generate_square_subsequent_mask(T).to(x.device)
            self.register_buffer('causal_mask', mask, persistent=False)

        out = self.transformer(x, mask=self.causal_mask, is_causal=True)
        return self.output_proj(out)

class CausalGraphLayer(nn.Module):
    def __init__(self, N, d_model, num_bases=4, max_k=32):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.num_bases = min(num_bases, d_model)
        self.max_k = max_k

        self.adjacency = nn.Parameter(torch.ones(N, self.max_k) * 0.5 + torch.randn(N, self.max_k) * 0.01)
        self.basis_weights = nn.Parameter(torch.randn(self.num_bases, N, self.max_k) * 0.02)
        self.channel_coeffs = nn.Parameter(torch.randn(d_model, self.num_bases))

        self.register_buffer('last_neighbor_indices', None)

    def forward(self, z, neighbor_indices):
        B, N, T, C = z.shape
        k_curr = neighbor_indices.size(1)

        self.last_neighbor_indices = neighbor_indices

        bases = self.basis_weights[:, :, :k_curr]
        adj = self.adjacency[:, :k_curr]

        eff_weights = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
        edge_weights = eff_weights * adj.unsqueeze(0)

        flat_indices = neighbor_indices.view(-1)
        z_flat = z.view(B, N, -1)
        z_neigh_flat = z_flat[:, flat_indices, :]
        z_neigh = z_neigh_flat.view(B, N, k_curr, T, C)

        w_aligned = edge_weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(3)
        z_out = (z_neigh * w_aligned).sum(dim=2)

        return torch.tanh(z_out)

    def get_soft_graph(self):
        if self.last_neighbor_indices is None:
            return None

        with torch.no_grad():
            indices = self.last_neighbor_indices
            k = indices.size(1)
            bases = self.basis_weights[..., :k]
            W = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
            W_mag = W.abs().mean(dim=0)
            sparse_w = self.adjacency[:, :k].abs() * W_mag

            dense = torch.zeros(self.N, self.N, device=self.adjacency.device)
            dense.scatter_(1, indices, sparse_w)
            return dense.T

    def structural_l1_loss(self):
        return torch.sum(torch.abs(self.adjacency))

class GeoDCDLayer(nn.Module):
    def __init__(self, N, d_model=64, nhead=4, num_layers=2, num_bases=4, max_k=100):
        super().__init__()
        self.geo_encoder = CausalTransformerBlock(1, d_model, d_model, nhead, num_layers)
        self.pred_head = nn.Linear(d_model, 1)
        self.graph = CausalGraphLayer(N, d_model, num_bases, max_k)

    def forward(self, x, mask):
        B, N, T = x.shape
        z = self.geo_encoder(x.reshape(B*N, T, 1)).view(B, N, T, -1)
        zhat_next = self.graph(z, neighbor_indices=mask)
        x_pred = self.pred_head(
            zhat_next[..., :-1, :].squeeze(-1)
        ).view(B, N, T-1)
        return x_pred, z

class GeometricPooler(nn.Module):
    def __init__(self, num_patches, shift_scale=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.shift_scale = shift_scale
        self.register_buffer('S_matrix', None)

    def get_structure(self, coords, training=False):
        coords_np = coords.detach().float().cpu().numpy()
        c_mean = coords_np.mean(axis=0)
        c_std = coords_np.std(axis=0) + 1e-5
        coords_norm = (coords_np - c_mean) / c_std

        if training:
            shift = np.random.randn(*coords_norm.shape) * self.shift_scale
            coords_norm += shift

        kmeans = KMeans(n_clusters=self.num_patches, random_state=None)
        labels = kmeans.fit_predict(coords_norm)

        N = coords.shape[0]
        S_hard = torch.zeros(N, self.num_patches, device=coords.device)
        labels_tensor = torch.tensor(labels, device=coords.device).long()
        S_hard.scatter_(1, labels_tensor.unsqueeze(1), 1.0)

        return S_hard

    def forward(self, x, coords):
        if self.training:
            S = self.get_structure(coords, training=True)
            self.S_matrix = S
            return S.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            if self.S_matrix is None:
                self.S_matrix = self.get_structure(coords, training=False)
            return self.S_matrix.unsqueeze(0).expand(x.shape[0], -1, -1)

class GeoDCD(nn.Module):
    def __init__(self, N, coords, hierarchy=[32, 8], d_model=64, num_bases=4, penalty_factor=5.0, max_k=100):
        super().__init__()
        self.dims = [N] + hierarchy
        self.num_levels = len(self.dims)
        self.penalty_factor = penalty_factor
        self.max_k = max_k

        coords = torch.tensor(coords).float() if not torch.is_tensor(coords) else coords
        self.register_buffer('coords', coords)

        self.layers = nn.ModuleList()
        self.poolers = nn.ModuleList()

        for i in range(self.num_levels):
            self.layers.append(
                GeoDCDLayer(self.dims[i], d_model, num_bases=num_bases, max_k=self.max_k)
            )
            if i < self.num_levels - 1:
                self.poolers.append(GeometricPooler(self.dims[i+1]))
                self.register_buffer(f'structure_S_{i}', torch.zeros(self.dims[i], self.dims[i+1]))
    
    def get_structural_l1_loss(self):
        return sum(layer.graph.structural_l1_loss() for layer in self.layers)

    def _get_knn_indices(self, coords, coarse_graph=None, structure_S=None, max_k=None):
        if max_k is None:
            max_k = self.max_k
        dists = torch.cdist(coords, coords)

        if (coarse_graph is not None) and (structure_S is not None):
            parents = structure_S.argmax(dim=1)
            coarse_bin = (coarse_graph > 0.001).float()
            prior_mask = coarse_bin[parents][:, parents]

            same_parent = (parents.unsqueeze(1) == parents.unsqueeze(0)).float()
            prior_mask = torch.max(prior_mask, same_parent)

            penalty = 1.0 + self.penalty_factor * (1.0 - prior_mask)
            dists = dists * penalty

        if self.training and coarse_graph is None:
             dists = dists + torch.randn_like(dists) * 0.01

        # Use threshold to filter neighbors
        threshold = dists.mean() + self.penalty_factor * dists.std()
        dists = torch.where(dists <= threshold, dists, torch.tensor(float('inf'), device=dists.device))

        k = min(max_k, coords.shape[0])
        _, indices = torch.topk(dists, k, dim=1, largest=False)
        return indices

    def forward(self, x):
        # Stage 1: Bottom-Up Aggregation
        xs, curr_coords = [x], self.coords
        coords_list, S_list = [curr_coords], []
        
        for i in range(self.num_levels - 1):
            S = self.poolers[i](xs[-1], curr_coords)
            S_list.append(S[0]) 
            with torch.no_grad(): getattr(self, f'structure_S_{i}').copy_(S[0])
            
            S_norm = S[0] / (S[0].sum(0, keepdim=True) + 1e-6)
            x_next = torch.matmul(xs[-1].permute(0,2,1), S_norm).permute(0,2,1)
            xs.append(x_next)
            
            curr_coords = torch.mm(S_norm.t(), curr_coords)
            coords_list.append(curr_coords)
            
        # Stage 2: Top-Down Causal Discovery
        results_dict = {}
        upper_level_graph = None

        for i in reversed(range(self.num_levels)):
            current_S = S_list[i] if i < len(S_list) else None

            mask = self._get_knn_indices(
                coords=coords_list[i],
                coarse_graph=upper_level_graph,
                structure_S=current_S,
                max_k=self.layers[i].graph.max_k
            )

            x_pred, _ = self.layers[i](xs[i], mask=mask)
            upper_level_graph = self.layers[i].graph.get_soft_graph()

            results_dict[i] = {
                'level': i,
                'x_pred': x_pred,
                'x_target': xs[i],
                'S': current_S,
                'k_used': coords_list[i].shape[0]
            }
            
        return [results_dict[i] for i in range(self.num_levels)]

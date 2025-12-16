import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.cluster import KMeans

# ==========================================
# 1. Basic Components
# ==========================================

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
        return x + self.pe[:, :x.size(1), :]

class CausalTransformerBlock(nn.Module):
    """Transformer Encoder with Causal Mask caching."""
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

# ==========================================
# 2. Core Graph Layer
# ==========================================

class CausalGraphLayer(nn.Module):
    """Sparse Causal Graph Layer (O(Nk)) with basis decomposition."""
    def __init__(self, N, d_model, max_k=32, num_bases=4):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.num_bases = min(num_bases, d_model)
        self.max_k = max_k

        # Params: Structure (N, max_k), Bases (K, N, max_k), Coeffs (C, K)
        self.adjacency = nn.Parameter(torch.ones(N, max_k) + torch.randn(N, max_k) * 0.01)
        self.basis_weights = nn.Parameter(torch.randn(self.num_bases, N, max_k) * 0.02)
        self.channel_coeffs = nn.Parameter(torch.randn(d_model, self.num_bases))
        
        self.register_buffer('last_neighbor_indices', None)

    def forward(self, z, neighbor_indices):
        """z: (B, N, T, C), neighbor_indices: (N, k_curr)"""
        B, N, T, C = z.shape
        k_curr = neighbor_indices.size(1)
        
        if k_curr > self.max_k: raise ValueError(f"k ({k_curr}) > max_k ({self.max_k})")
        
        # Cache for visualization
        self.last_neighbor_indices = neighbor_indices

        # Dynamic slicing
        bases = self.basis_weights[:, :, :k_curr]
        adj = self.adjacency[:, :k_curr]
        
        # W_eff = Coeffs @ Bases -> (C, N, k)
        eff_weights = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
        edge_weights = eff_weights * adj.unsqueeze(0)
        
        # Gather neighbors: (B, N, k, T, C)
        z_neigh = z[:, neighbor_indices]
        
        # Weighted Sum: weights (1, N, k, 1, C) * z_neigh
        w_aligned = edge_weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(3)
        z_out = (z_neigh * w_aligned).sum(dim=2)
        
        return torch.tanh(z_out)

    def get_soft_graph(self):
        """Reconstruct dense NxN adjacency for visualization."""
        if self.last_neighbor_indices is None: return None
        
        with torch.no_grad():
            indices = self.last_neighbor_indices
            k = indices.size(1)
            
            # Recompute scalar weight magnitude: (N, k)
            bases = self.basis_weights[..., :k]
            W = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
            W_mag = W.abs().mean(dim=0)
            sparse_w = self.adjacency[:, :k].abs() * W_mag
            
            # Scatter to dense
            dense = torch.zeros(self.N, self.N, device=self.adjacency.device)
            dense.scatter_(1, indices, sparse_w)
            return dense.t()

    def structural_l1_loss(self):
        return torch.sum(torch.abs(self.adjacency))

# ==========================================
# 3. GeoDCD Layer
# ==========================================

class GeoDCDLayer(nn.Module):
    """Combines Temporal (Transformer) and Spatial (Graph) modeling."""
    def __init__(self, N, d_model=64, nhead=4, num_layers=2, num_bases=4, max_k=32):
        super().__init__()
        self.geo_encoder = CausalTransformerBlock(1, d_model, d_model, nhead, num_layers)
        self.pred_head = nn.Linear(d_model, 1)
        self.graph = CausalGraphLayer(N, d_model, max_k, num_bases)

    def forward(self, x, mask):
        B, N, T = x.shape
        # Temporal Encode -> z
        z = self.geo_encoder(x.reshape(B*N, T, 1)).view(B, N, T, -1)
        
        # Spatial Dynamics: z_t -> z_{t+1}
        zhat_next = self.graph(z, neighbor_indices=mask)
        
        # Prediction: z_{t+1} -> x_{t+1} (Causal slicing [:-1])
        x_pred = self.pred_head(
            zhat_next[..., :-1, :].squeeze(-1)
        ).view(B, N, T-1)
        
        return x_pred, z

# ==========================================
# 4. Pooling Layer
# ==========================================

class GeometricPooler(nn.Module):
    """Physical coordinate pooling using K-Means."""
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.register_buffer('S_matrix', None)
        self.initialized = False

    def init_structure(self, coords):
        if self.initialized: return
        
        # Coordinate normalization
        coords_np = coords.detach().float().cpu().numpy()
        c_mean = coords_np.mean(axis=0)
        c_std = coords_np.std(axis=0) + 1e-5
        coords_norm = (coords_np - c_mean) / c_std
        
        # K-Means & Hard Assignment
        kmeans = KMeans(n_clusters=self.num_patches, n_init=20, random_state=42)
        labels = kmeans.fit_predict(coords_norm)
        
        # Generate Hard Assignment S (N, K)
        N = coords.shape[0]
        S_hard = torch.zeros(N, self.num_patches, device=coords.device)
        labels_tensor = torch.tensor(labels, device=coords.device).long()
        S_hard.scatter_(1, labels_tensor.unsqueeze(1), 1.0)
        
        self.S_matrix = S_hard
        self.initialized = True
        print(f"🔲 Geometric Window Initialized: N={N} -> K={self.num_patches}")

    def forward(self, x, coords):
        if not self.initialized: self.init_structure(coords)
        return self.S_matrix.unsqueeze(0).expand(x.shape[0], -1, -1)

# ==========================================
# 5. Main Model
# ==========================================

class GeoDCD(nn.Module):
    def __init__(self, N, coords, hierarchy=[32, 8], d_model=64, num_bases=4, max_k=32):
        super().__init__()
        self.dims = [N] + hierarchy
        self.num_levels = len(self.dims)
        self.max_k = max_k
        
        coords = torch.tensor(coords).float() if not torch.is_tensor(coords) else coords
        self.register_buffer('coords', coords)
        
        self.layers = nn.ModuleList()
        self.poolers = nn.ModuleList()
        
        for i in range(self.num_levels):
            # Graph Layer
            self.layers.append(
                GeoDCDLayer(self.dims[i], d_model, num_bases=num_bases, max_k=self.max_k)
            )
            # Pooler (except last)
            if i < self.num_levels - 1:
                self.poolers.append(GeometricPooler(self.dims[i+1]))
                self.register_buffer(f'structure_S_{i}', torch.zeros(self.dims[i], self.dims[i+1]))
    
    def get_structural_l1_loss(self):
        return sum(layer.graph.structural_l1_loss() for layer in self.layers)

    def _get_knn_indices(self, coords, k, shift=False):
        if shift and self.training:
            coords = coords + torch.randn_like(coords) * 0.05 * coords.std(0, keepdim=True)
            
        dists = torch.cdist(coords, coords)
        _, indices = torch.topk(dists, min(k, coords.shape[0]), dim=1, largest=False)
        return indices

    def forward(self, x, tau=1.0):
        xs, curr_coords = [x], self.coords
        coords_list, S_list = [curr_coords], []
        
        # --- Bottom-up: Aggregation ---
        for i in range(self.num_levels - 1):
            S = self.poolers[i](xs[-1], curr_coords)
            S_list.append(S)
            with torch.no_grad(): getattr(self, f'structure_S_{i}').copy_(S[0])
            
            # Aggregate X and Coords
            S_norm = S[0] / (S[0].sum(0, keepdim=True) + 1e-6)
            xs.append(torch.matmul(xs[-1].permute(0,2,1), S_norm).permute(0,2,1))
            curr_coords = torch.mm(S_norm.t(), curr_coords)
            coords_list.append(curr_coords)
            
        # --- Multi-scale Causal Discovery ---
        results = []
        for i in range(self.num_levels):
            target_k = min(self.max_k, self.dims[i])
            mask = self._get_knn_indices(coords_list[i], target_k, shift=(i > 0))
            
            x_pred, _ = self.layers[i](xs[i], mask=mask)
            
            results.append({
                'level': i,
                'x_pred': x_pred,
                'x_target': xs[i],
                'S': S_list[i] if i < len(S_list) else None,
                'k_used': target_k
            })
            
        return results
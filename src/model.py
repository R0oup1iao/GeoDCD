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
        self.adjacency = nn.Parameter(torch.ones(N, max_k) * 0.5 + torch.randn(N, max_k) * 0.01)
        self.basis_weights = nn.Parameter(torch.randn(self.num_bases, N, max_k) * 0.02)
        self.channel_coeffs = nn.Parameter(torch.randn(d_model, self.num_bases))
        
        self.register_buffer('last_neighbor_indices', None)

    def forward(self, z, neighbor_indices):
        """z: (B, N, T, C), neighbor_indices: (N, k_curr)"""
        B, N, T, C = z.shape
        k_curr = neighbor_indices.size(1)
        
        # Cache for visualization
        self.last_neighbor_indices = neighbor_indices

        if k_curr > self.max_k: 
            neighbor_indices = neighbor_indices[:, :self.max_k]
            k_curr = self.max_k
        
        # Dynamic slicing
        bases = self.basis_weights[:, :, :k_curr]
        adj = self.adjacency[:, :k_curr]
        
        # W_eff = Coeffs @ Bases -> (C, N, k)
        eff_weights = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
        edge_weights = eff_weights * adj.unsqueeze(0)
        
        # Gather neighbors using fancy indexing
        flat_indices = neighbor_indices.view(-1) # (N*k)
        z_flat = z.view(B, N, -1) # (B, N, T*C)
        z_neigh_flat = z_flat[:, flat_indices, :]
        z_neigh = z_neigh_flat.view(B, N, k_curr, T, C)
        
        # Weighted Sum
        w_aligned = edge_weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(3)
        z_out = (z_neigh * w_aligned).sum(dim=2)
        
        return torch.tanh(z_out)

    def get_soft_graph(self):
        """Reconstruct dense NxN adjacency."""
        if self.last_neighbor_indices is None: return None
        
        with torch.no_grad():
            indices = self.last_neighbor_indices
            k = indices.size(1)
            bases = self.basis_weights[..., :k]
            W = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
            W_mag = W.abs().mean(dim=0)
            sparse_w = self.adjacency[:, :k].abs() * W_mag
            
            dense = torch.zeros(self.N, self.N, device=self.adjacency.device)
            dense.scatter_(1, indices, sparse_w)
            return dense

    def structural_l1_loss(self):
        return torch.sum(torch.abs(self.adjacency))

# ==========================================
# 3. GeoDCD Layer
# ==========================================

class GeoDCDLayer(nn.Module):
    def __init__(self, N, d_model=64, nhead=4, num_layers=2, num_bases=4, max_k=32):
        super().__init__()
        self.geo_encoder = CausalTransformerBlock(1, d_model, d_model, nhead, num_layers)
        self.decoder = CausalTransformerBlock(d_model, 1, d_model, nhead, num_layers)
        self.graph = CausalGraphLayer(N, d_model, max_k, num_bases)

    def forward(self, x, mask):
        B, N, T = x.shape
        z = self.geo_encoder(x.reshape(B*N, T, 1)).view(B, N, T, -1)
        x_recon = self.decoder(z.view(B*N, T, -1)).view(B, N, T)
        zhat_next = self.graph(z, neighbor_indices=mask)
        x_pred = self.decoder(
            zhat_next[..., :-1, :].contiguous().view(B*N, T-1, -1)
        ).view(B, N, T-1)
        return x_recon, x_pred, z

# ==========================================
# 4. Pooling Layer
# ==========================================

class GeometricPooler(nn.Module):
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.register_buffer('S_matrix', None)
        self.initialized = False

    def init_structure(self, coords):
        if self.initialized: return
        coords_np = coords.detach().float().cpu().numpy()
        c_mean = coords_np.mean(axis=0)
        c_std = coords_np.std(axis=0) + 1e-5
        coords_norm = (coords_np - c_mean) / c_std
        kmeans = KMeans(n_clusters=self.num_patches, n_init=20, random_state=42)
        labels = kmeans.fit_predict(coords_norm)
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
# 5. Main Model (Soft Constrained Top-Down)
# ==========================================

class GeoDCD(nn.Module):
    def __init__(self, N, coords, hierarchy=[32, 8], d_model=64, num_bases=4):
        super().__init__()
        self.dims = [N] + hierarchy
        self.num_levels = len(self.dims)
        self.max_k = 32 
        
        # Soft Penalty Factor: 
        # A value of 5.0 means unconnected parent regions are effectively 
        # "pushed away" by 6x their original distance.
        # This preserves local neighbors (0.1 * 6 = 0.6, still close)
        # but suppresses far ones (2.0 * 6 = 12.0, pushed out of TopK)
        self.penalty_factor = 5.0 
        
        coords = torch.tensor(coords).float() if not torch.is_tensor(coords) else coords
        self.register_buffer('coords', coords)
        
        self.layers = nn.ModuleList()
        self.poolers = nn.ModuleList()
        
        for i in range(self.num_levels):
            current_max_k = min(self.max_k, self.dims[i])
            self.layers.append(
                GeoDCDLayer(self.dims[i], d_model, num_bases=num_bases, max_k=current_max_k)
            )
            if i < self.num_levels - 1:
                self.poolers.append(GeometricPooler(self.dims[i+1]))
                self.register_buffer(f'structure_S_{i}', torch.zeros(self.dims[i], self.dims[i+1]))
    
    def get_structural_l1_loss(self):
        return sum(layer.graph.structural_l1_loss() for layer in self.layers)

    def _get_knn_indices(self, coords, k, coarse_graph=None, structure_S=None):
        # 1. Physical Distances
        dists = torch.cdist(coords, coords)
        
        # 2. Top-Down Soft Constraint
        if (coarse_graph is not None) and (structure_S is not None):
            parents = structure_S.argmax(dim=1)
            # Use lower threshold to capture weak signals
            coarse_bin = (coarse_graph > 0.001).float() 
            prior_mask = coarse_bin[parents][:, parents]
            
            # Ensure self-block connectivity
            same_parent = (parents.unsqueeze(1) == parents.unsqueeze(0)).float()
            prior_mask = torch.max(prior_mask, same_parent)
            
            # Apply Multiplicative Penalty
            # D_new = D_old * (1 + beta * (1 - Mask))
            penalty = 1.0 + self.penalty_factor * (1.0 - prior_mask)
            dists = dists * penalty

        # 3. Select Top-K
        if self.training and coarse_graph is None:
             dists = dists + torch.randn_like(dists) * 0.01
             
        k = min(k, coords.shape[0])
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
            target_k = min(self.max_k, self.dims[i])
            
            mask = self._get_knn_indices(
                coords=coords_list[i], 
                k=target_k, 
                coarse_graph=upper_level_graph, 
                structure_S=current_S
            )
            
            x_rec, x_pred, _ = self.layers[i](xs[i], mask=mask)
            upper_level_graph = self.layers[i].graph.get_soft_graph()
            
            results_dict[i] = {
                'level': i,
                'x_rec': x_rec,
                'x_pred': x_pred,
                'x_target': xs[i],
                'S': current_S,
                'k_used': target_k
            }
            
        return [results_dict[i] for i in range(self.num_levels)]
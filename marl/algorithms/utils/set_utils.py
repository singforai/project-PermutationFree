import math 
import torch

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

class SetNorm(nn.LayerNorm):
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x):
        # standardization
        out = super().forward(x)
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, sample_size, v_norm_samples = 32, normalizeQ=True):
        super().__init__()
        self.num_heads = num_heads
        self.normalizeQ = normalizeQ
        self.hidden_size = hidden_size
        self.dim_split = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0
    
        self.project_queries = nn.Linear(self.hidden_size, self.hidden_size)
        self.project_keys = nn.Linear(self.hidden_size, self.hidden_size)
        self.project_values = nn.Linear(self.hidden_size, self.hidden_size)
        self.concatenation = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.sample_size = sample_size
        self.softmax = nn.Softmax(dim=1)

        self.normQ = SetNorm(
            [self.sample_size, self.hidden_size], elementwise_affine=False, feature_dim=self.hidden_size
        )
        self.normK = SetNorm(
            [v_norm_samples, self.hidden_size], elementwise_affine=False, feature_dim=self.hidden_size
        )
        self.norm0 = SetNorm(
            [self.sample_size, self.hidden_size], elementwise_affine=False, feature_dim=self.hidden_size
        )
        
    def forward(self, Q, K):
        _input = Q
        if self.normalizeQ:
            Q_norm = self.normQ(Q)
        K_norm = self.normK(K)

        proj_Q = self.project_queries(Q_norm)
        proj_K = self.project_keys(K_norm)
        proj_V = self.project_values(K_norm)
        
        dim_split = 2**int(round(np.log2(self.dim_split),0))
        
        Q_ = torch.cat(proj_Q.split(dim_split, 2), 0)
        K_ = torch.cat(proj_K.split(dim_split, 2), 0)
        V_ = torch.cat(proj_V.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.hidden_size), 2)
        input_multihead = torch.cat(_input.split(dim_split, 2), 0)
        O = torch.cat((input_multihead + A.bmm(V_)).split(Q_norm.size(0), 0), 2)
        
        normed_O = self.norm0(O)
        O = O + F.relu(self.concatenation(normed_O))
        return O
    
class SAB(nn.Module):
    def __init__(self, hidden_size, num_heads, sample_size):
        super().__init__()
        self.mab = MultiheadAttentionBlock(
            hidden_size = hidden_size, 
            num_heads = num_heads, 
            sample_size=sample_size,
            v_norm_samples = sample_size
        )
    def forward(self, X):
        return self.mab(X, X)
        
class PMA(nn.Module):
    def __init__(self, hidden_size, num_seed_vector, num_heads, v_norm_samples):
        super().__init__()
        self.seed_vectors = nn.init.xavier_uniform_(
            nn.Parameter(torch.randn(1, num_seed_vector, hidden_size))
        )
        self.mab = MultiheadAttentionBlock(
            hidden_size = hidden_size, 
            num_heads = num_heads, 
            sample_size=num_seed_vector, 
            v_norm_samples = v_norm_samples
        )

    def forward(self, x):
        return self.mab(self.seed_vectors.repeat(x.size(0), 1, 1), x)
import torch
import torch.nn as nn
from math import sqrt
import numpy as np

class Dynamic_Directional_Attention(nn.Module):
    def __init__(self, p, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(Dynamic_Directional_Attention, self).__init__()
        self.scale = scale 
        self.mask_flag = mask_flag  
        self.output_attention = output_attention  
        self.dropout = nn.Dropout(attention_dropout) 
        self.directional_weights = nn.Parameter(torch.ones((1, 1, 1, 1)), requires_grad=True)  # learnable weight
        self.dynamic_param = nn.Parameter(torch.randn(1), requires_grad=True)  # learnable dynamic parameter
        self.p = p
        
    def directional_reweighting(self, x):
        return x * self.directional_weights  # [B, L/S, H, E]

    def axis_aligned_transformation(self, x, p):
        #  [B, L/S, H, E]
        return x * (1 / (torch.std(x, dim=-2, keepdim=True) + 1e-6)).pow(p)

    def nonlinear_mapping(self, x):
        return torch.tanh(x) * self.dynamic_param

    def compute_tau(self, scores):
        score_var = torch.var(scores, dim=-1, keepdim=True)
        return torch.sqrt(score_var + 1e-6)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  
        _, S, _ ,D= keys.shape  
        scale = self.scale or 1. / sqrt(E) 
        
        queries_transformed = self.nonlinear_mapping(
            self.directional_reweighting(
                self.axis_aligned_transformation(queries,self.p)
            )
        )  # [B, L, H, E]

        keys_transformed = self.nonlinear_mapping(
            self.directional_reweighting(
                self.axis_aligned_transformation(keys,self.p)
            )
        )  # [B, S, H, E]

        scores = torch.einsum("blhe,bshe->bhls", queries_transformed, keys_transformed)  #  [B, H, L, S]

        if self.mask_flag:
            scores.masked_fill_(attn_mask, -np.inf)

        tau = self.compute_tau(scores) if tau is None else tau

        A = self.dropout(torch.softmax(scale * scores / tau, dim=-1))  # [B, H, L, S]

        V = torch.einsum("bhls,bshd->blhd", A, values)  # [B, L, H, D]

        if self.output_attention:
            return V.contiguous(), A  
        else:
            return V.contiguous(), None 

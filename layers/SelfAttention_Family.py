import torch
import torch.nn as nn
import numpy as np
import math,os
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
from torch.nn.functional import softmax
from torch.autograd.functional import jacobian
import torch.nn.functional as F
from torch.nn.functional import relu

class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    # 初始化函数，定义了该类的基本属性。
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()  # 调用父类nn.Module的初始化函数。
        self.scale = scale  # 缩放因子，如果不提供，则在前向传播时会设置为1/sqrt(E)。
        self.mask_flag = mask_flag  # 一个标志，指示是否使用掩码来避免在计算注意力时看到未来的信息。
        self.output_attention = output_attention  # 一个标志，指示是否输出注意力矩阵A。
        self.dropout = nn.Dropout(attention_dropout)  # 定义一个dropout层，用于在注意力权重上进行dropout操作。
    # 前向传播函数，定义了该模块的计算逻辑。
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # 从queries获取批次大小B，序列长度L，头数H和每头维度E。
        _, S, _, D = values.shape  # 从values获取序列长度S和每头维度D，通常S和L相同，D和E相同。
        scale = self.scale or 1. / sqrt(E)  # 如果没有给定scale，则使用1/sqrt(E)作为缩放因子。
        # 使用爱因斯坦求和约定进行批量矩阵乘法，计算所有头的查询和键的分数。
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        dir_path = '/home/zhouziyu/Time-Series-Library-main/adata'
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, 'patchtst.pt')
        torch.save(scores, file_path)
        
        # 如果设置了掩码标志，将使用掩码来修改分数。
        if self.mask_flag:
            if attn_mask is None:
                # 如果没有提供attn_mask，则创建一个三角形因果掩码。
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # 使用掩码更新分数，掩码位置的分数设置为负无穷，这将在softmax后变为0，相当于忽略这些位置。
            scores.masked_fill_(attn_mask.mask, -np.inf)
        # 应用softmax函数并缩放分数，然后应用dropout。
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 再次使用爱因斯坦求和约定，根据注意力权重A和值矩阵values计算输出。
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # 根据output_attention标志决定输出。
        if self.output_attention:
            return V.contiguous(), A  # 如果需要输出注意力矩阵，则返回它和V。
        else:
            return V.contiguous(), None  # 否则只返回V，注意力矩阵为None。
          






class Dynamic_Directional_Attention(nn.Module):
    def __init__(self, p, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(Dynamic_Directional_Attention, self).__init__()
        self.scale = scale  # 缩放因子，用于调节查询和键的点积结果
        self.mask_flag = mask_flag  # 是否使用掩码来处理变长序列
        self.output_attention = output_attention  # 是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # 注意力权重的dropout处理
        self.directional_weights = nn.Parameter(torch.ones((1, 1, 1, 1)), requires_grad=True)  # 可学习的方向性权重
        self.dynamic_param = nn.Parameter(torch.randn(1), requires_grad=True)  # 可学习的动态参数
        self.p = p
    def directional_reweighting(self, x):
        # 对输入应用方向性权重
        return x * self.directional_weights  # 形状: [B, L/S, H, E]

    def axis_aligned_transformation(self, x, p):
        # 轴对齐变换，用于标准化不同维度的尺度
        # 形状: [B, L/S, H, E]
        return x * (1 / (torch.std(x, dim=-2, keepdim=True) + 1e-6)).pow(p)

    def nonlinear_mapping(self, x):
        # 应用非线性映射，增加模型的表达能力
        # 形状: [B, L/S, H, E]
        return torch.tanh(x) * self.dynamic_param

    def compute_tau(self, scores):
        # 根据分数的变异性动态计算tau
        # 形状: [B, H, L, S]
        score_var = torch.var(scores, dim=-1, keepdim=True)
        return torch.sqrt(score_var + 1e-6)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # 提取查询的维度信息
        _, S, _ ,D= keys.shape  # 提取键的维度信息
        scale = self.scale or 1. / sqrt(E)  # 如果未提供scale，则默认使用1/sqrt(E)

        # 对查询和键进行转换处理
        queries_transformed = self.nonlinear_mapping(
            self.directional_reweighting(
                self.axis_aligned_transformation(queries,self.p)
            )
        )  # 形状: [B, L, H, E]

        keys_transformed = self.nonlinear_mapping(
            self.directional_reweighting(
                self.axis_aligned_transformation(keys,self.p)
            )
        )  # 形状: [B, S, H, E]

        # 计算点积得分
        scores = torch.einsum("blhe,bshe->bhls", queries_transformed, keys_transformed)  # 形状: [B, H, L, S]

        if self.mask_flag:
            # 如果启用掩码，处理变长序列
            scores.masked_fill_(attn_mask, -np.inf)

        # 动态计算tau
        tau = self.compute_tau(scores) if tau is None else tau

        # 应用Softmax函数得到注意力权重
        A = self.dropout(torch.softmax(scale * scores / tau, dim=-1))  # 形状: [B, H, L, S]

        # 将注意力权重应用于值
        V = torch.einsum("bhls,bshd->blhd", A, values)  # 形状: [B, L, H, D]

        if self.output_attention:
            return V.contiguous(), A  # 返回输出和注意力权重
        else:
            return V.contiguous(), None  # 只返回输出


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

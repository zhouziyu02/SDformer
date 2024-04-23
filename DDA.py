import torch
import torch.nn as nn
from math import sqrt
import numpy as np

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
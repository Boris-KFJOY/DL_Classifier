# -*- coding: utf-8 -*-
# SSVEPformerX: subband + cross-view + harmonic PE, drop-in replacement
# Safe to co-exist with the original SSVEPformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ----- Utils -----
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x.contiguous(), **kwargs)

class FFN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):  # (B, T, D)
        return self.net(x)

class TinyTransformer(nn.Module):
    def __init__(self, depth: int, token_num: int, token_dim: int, ksize: int = 7, dropout: float = 0.1):
        """Conv-style attention block，与原模型保持风格一致（1D conv沿频域）"""
        super().__init__()
        layers = []

        for _ in range(depth):
            layers += [
                PreNorm(token_dim, nn.Sequential(
                    nn.Conv1d(token_num, token_num, kernel_size=ksize, padding=ksize//2, groups=1),
                    nn.LayerNorm(token_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )),
                PreNorm(token_dim, FFN(token_dim, dropout))
            ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):  # x: (B, T=token_num, D=token_dim)
        # 将 (B,T,D) 换到 (B,T,D) → conv 需要 (B,T,D) 先转为 (B,T,D) -> (B,T,D)
        # 这里我们把 conv 放在 token 维度上: 需要 (B,T,D)->(B,T,D) 先转成 (B,T,D)

        for (attn, ffn) in zip(self.layers[0::2], self.layers[1::2]):
            x = x + attn(x)
            x = x + ffn(x)
        return x

# ----- New ideas -----
class HarmonicPE(nn.Module):
    """可学习的频域/谐波位置编码：shape = (1, token_num, token_dim)"""
    def __init__(self, token_num: int, token_dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, token_num, token_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)
    def forward(self, x):  # x: (B,T,D)
        return x + self.pe

class SubBandGate(nn.Module):
    """
    子带频域建模：对 (B, T, D) 的 D 维做多尺度卷积，模拟滤波器组 → 加权融合
    等价于在频域上建多个子带视图 (K 个)，然后门控加权到主分支。
    """
    def __init__(self, token_num: int, token_dim: int, k_list=(5, 9, 17), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(token_num, token_num, kernel_size=k, padding=k//2, groups=token_num),  # depthwise
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k in k_list
        ])
        self.proj = nn.Linear(token_dim, token_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(len(k_list)))
        nn.init.zeros_(self.gate)

    def forward(self, x):  # x: (B, T, D)
        # 目标：沿 D 维做子带卷积（长度维 = D，通道 = T），然后在最后一维做线性投影
        # 当前 self.branches 里每个分支是 Conv1d(in_channels=token_num=T, out_channels=T, kernel=k, padding=k//2, groups=T)
        # 因此它们期望输入就是 (B, C=T, L=D) —— 与 x 的 (B, T, D) 完全一致，无需转置
        x_in = x
        x = x.contiguous()

        outs = []
        for b in self.branches:
            o = b(x)  # 仍是 (B, T, D)
            outs.append(o)

        g = torch.softmax(self.gate, dim=0)  # (K,)
        # 门控加权融合，保持 (B, T, D)
        y = torch.zeros_like(x)
        for w, o in zip(g, outs):
            y = y + w * o

        # 线性层在最后一维 D 上做投影：保持 (B, T, D)
        y = self.proj(y)

        return x_in + y

class CrossViewBlock(nn.Module):
    """
    轻量双视图交叉注意：把输入投影成两种视图 A/B，做一次 multihead attention 交叉融合。
    这里用 PyTorch 自带 MHA；为了简单稳定，embed_dim = token_dim。
    """
    def __init__(self, token_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj_a = nn.Linear(token_dim, token_dim, bias=False)
        self.proj_b = nn.Linear(token_dim, token_dim, bias=False)
        self.mha_ab = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_ba = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = FFN(token_dim, dropout)
        self.ln = nn.LayerNorm(token_dim)

    def forward(self, x):  # (B, T, D)
        a = self.proj_a(x)
        b = self.proj_b(x)
        # A attends to B, and B attends to A
        a2, _ = self.mha_ab(query=a, key=b, value=b, need_weights=False)
        b2, _ = self.mha_ba(query=b, key=a, value=a, need_weights=False)
        y = 0.5 * (a2 + b2)
        y = self.ln(x + y)
        y = y + self.ffn(y)
        return y

# ----- The drop-in model -----
class SSVEPformerX(nn.Module):
    """
    输入/输出张量形状与原 SSVEPformer 完全一致：
      input:  (B, chs_num, token_dim)  —— 你的预处理已经产出该形状
      output: (B, class_num)
    内部会把 (B, chs, D) 通过 1x1 conv 提升到 token_num=chs*2，与旧模型约定对齐。
    """
    def __init__(self,
                 depth: int,
                 attention_kernal_length: int,  # 仅用于保持参数接口一致
                 chs_num: int,
                 class_num: int,
                 token_dim: int,
                 dropout: float = 0.1,
                 use_subband: bool = True,
                 use_crossview: bool = True,
                 use_harmonic_pe: bool = True):
        super().__init__()

        token_num = chs_num * 2
        # 与旧模型对齐  (见你现有 SSVEPformer.py)
        self.token_num = token_num
        self.token_dim = token_dim

        # Patch embedding（与旧模型保持一致的 1x1 conv + LN + GELU）
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(chs_num, self.token_num, kernel_size=1, padding=0, groups=1),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 三个可选增强模块
        self.harmonic_pe = HarmonicPE(self.token_num, self.token_dim) if use_harmonic_pe else nn.Identity()
        self.subband = SubBandGate(self.token_num, self.token_dim, k_list=(5, 9, 17),
                                   dropout=dropout) if use_subband else nn.Identity()
        self.crossview = CrossViewBlock(self.token_dim, num_heads=4,
                                        dropout=dropout) if use_crossview else nn.Identity()
        # 主干：轻量 Transformer（与原风格一致）

        self.backbone = TinyTransformer(depth=depth, token_num=self.token_num,
                                        token_dim=self.token_dim, ksize=attention_kernal_length, dropout=dropout)

        # 分类头：简洁稳健（保留与旧模型相同的输出维度）
        hidden = class_num * 6
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim * self.token_num, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, class_num)
        )

        # 权重初始化（与旧模型一致）
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, chs_num, token_dim)
        # 与旧模型相同的 entry
        if x.size(1) != self.chs_num or x.size(-1) != self.token_dim:
            raise ValueError(f"SSVEPformerX expects (B,{self.chs_num},{self.token_dim}), "
                             f"but got {tuple(x.shape)}")
        x = self.to_patch_embedding(x)          # (B, token_num, token_dim) —— 已经是 "tokens × freq"
        x = self.harmonic_pe(x)                 # 加谐波位置编码

        x = self.subband(x)                     # 子带门控
        # backbones 统一以 (B,T,D) 形式工作

        x = self.backbone(x)
        # 轻量 conv-transformer

        x = self.crossview(x)                   # 双视图交叉注意（轻量）
        out = self.head(x)

        return out

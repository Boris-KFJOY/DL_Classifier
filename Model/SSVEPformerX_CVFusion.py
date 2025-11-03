# -*- coding: utf-8 -*-
# SSVEPformerX_CVFusion: frequency + time dual-branch fusion via CrossView
# Extends the original SSVEPformerX while keeping utilities compatible
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

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
    """
    三种位置编码：
    - harmonic: 正弦/余弦的谐波基（k=1..H），可用于SSVEP的谐波建模
    - fixed   : 标准Transformer的sin/cos（不可学习）
    - learned : 可学习表（nn.Parameter）
    输出形状: (1, T, D)，与 tokens 相加或拼接
    """
    def __init__(self,
                 token_num: int,
                 token_dim: int,
                 enable: bool = True,
                 beta: float = 1.0,
                 mode: str = "harmonic",
                 max_harmonics: Optional[int] = None):
        super().__init__()
        self.enable = enable
        self.beta = beta
        self.mode = mode
        self.token_num = token_num
        self.token_dim = token_dim

        # learned 模式下使用的可学习表
        self.pe_table = nn.Parameter(torch.zeros(1, token_num, token_dim))
        nn.init.trunc_normal_(self.pe_table, std=0.02)

        # harmonic 模式的角频率（k=1..H），H 默认取 D//2（因为有 sin+cos 两路）
        H = max_harmonics or max(1, token_dim // 2)
        self.register_buffer("omega", 2 * math.pi * torch.arange(1, H + 1, dtype=torch.float32))  # [H]

        # fixed 模式（标准 transformer）用的 div_term
        div_term = torch.exp(torch.arange(0, token_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / token_dim))
        self.register_buffer("div_term", div_term)  # [D//2]

    # ---------- 公共入口 ----------
    def forward(self, x_or_T=None):
        """
        x_or_T: int (序列长度T) 或 Tensor（从中推断 T 和 device）
        返回: (1, T, D)
        """
        T, device, dtype = self._resolve_T_device_dtype(x_or_T)

        if not self.enable:
            return self._zeros(T, device, dtype)

        if self.mode == "harmonic":
            pe = self._harmonic_pe(T, device, dtype)        # (1,T,D)
        elif self.mode == "fixed":
            pe = self._fixed_bank(T, device, dtype).detach() # (1,T,D)
        elif self.mode == "learned":
            pe = self.pe_table[:, :T, :]                    # (1,T,D)
        else:
            raise ValueError(f"Unknown PE mode: {self.mode}")

        return self.beta * pe

    # ---------- 实现细节 ----------
    def _resolve_T_device_dtype(self, x_or_T):
        if isinstance(x_or_T, int):
            T = x_or_T
            device = self.pe_table.device
            dtype = self.pe_table.dtype
        elif torch.is_tensor(x_or_T):
            # 允许传 (B,T,...) 或 (T,...)；统一从倒数第二维取 T
            T = x_or_T.shape[-2] if x_or_T.dim() >= 2 else x_or_T.shape[-1]
            device = x_or_T.device
            dtype = x_or_T.dtype
        else:
            T = self.token_num
            device = self.pe_table.device
            dtype = self.pe_table.dtype
        return T, device, dtype

    def _zeros(self, T, device, dtype):
        return torch.zeros(1, T, self.token_dim, device=device, dtype=dtype)

    def _time_grid(self, T, device, dtype):
        # 归一化时间/位置网格 [0,1)
        return torch.linspace(0, 1, steps=T, device=device, dtype=dtype)

    def _harmonic_pe(self, T, device, dtype):
        """
        使用谐波基：sin(2π k t), cos(2π k t), k=1..H
        产物拼成 (T, 2H)；如 2H < D 则右侧零填充，>D 则截断。
        """
        t = self._time_grid(T, device, dtype)               # [T]
        phase = t[:, None] * self.omega[None, :]            # [T,H]
        sinc = torch.sin(phase)
        cosc = torch.cos(phase)
        pe_tc = torch.cat([sinc, cosc], dim=-1)             # [T, 2H]

        # pad/截断到 D
        if pe_tc.shape[-1] < self.token_dim:
            pe_tc = F.pad(pe_tc, (0, self.token_dim - pe_tc.shape[-1]))
        elif pe_tc.shape[-1] > self.token_dim:
            pe_tc = pe_tc[:, :self.token_dim]

        return pe_tc.unsqueeze(0)                           # (1,T,D)

    def _fixed_bank(self, T, device, dtype):
        """
        标准 Transformer 的正弦位置编码（不可学习）
        """
        position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # [T,1]
        pe = torch.zeros(T, self.token_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * self.div_term[:pe[:, 0::2].shape[1]])
        if self.token_dim > 1:
            pe[:, 1::2] = torch.cos(position * self.div_term[:pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)  # (1,T,D)
class DualHead(nn.Module):
    def __init__(self, token_dim, hidden, class_num, pool='gap', dropout=0.5):
        super().__init__()
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, class_num),
        )

    def forward(self, x_t, x_f):  # (B, 2C, F), (B, 2C, F)
        if self.pool == 'gap':
            t_vec = x_t.mean(dim=1)   # (B, F)
            f_vec = x_f.mean(dim=1)   # (B, F)
        else:                         # 如果你有专门 CLS token
            t_vec = x_t[:, 0]
            f_vec = x_f[:, 0]

        feat = torch.cat([t_vec, f_vec], dim=-1)  # (B, 2F)
        return self.mlp(feat)
class SubBandGate(nn.Module):
    """
    子带频域建模：对 (B, T, D) 的 D 维做多尺度卷积，模拟滤波器组 → 加权融合
    等价于在频域上建多个子带视图 (K 个)，然后门控加权到主分支。
    """
    def __init__(self, token_num: int, token_dim: int, k_list=(5, 9, 17), dropout: float = 0.1,proj_type="linear", in_ch=None):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(token_num, token_num, kernel_size=k, padding=k//2, groups=token_num),  # depthwise
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k in k_list
        ])
        if proj_type == "identity":
            self.proj = nn.Identity()
        elif proj_type == "dwconv3":
            self.proj = nn.Conv1d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        else:  # "linear"（原始）
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

    def __init__(self, token_dim, num_heads=4, dropout=0.1, enable=True, alpha=1.0):
        super().__init__()
        self.q_t = nn.Linear(token_dim, token_dim, bias=False)
        self.kv_f = nn.Linear(token_dim, token_dim, bias=False)
        self.q_f = nn.Linear(token_dim, token_dim, bias=False)
        self.kv_t = nn.Linear(token_dim, token_dim, bias=False)

        self.mha_t2f = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)  # t<-f
        self.mha_f2t = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)  # f<-t

        self.ffn_t = FFN(token_dim, dropout)
        self.ffn_f = FFN(token_dim, dropout)
        self.ln_t = nn.LayerNorm(token_dim)
        self.ln_f = nn.LayerNorm(token_dim)

        self.enable = enable
        self.alpha = alpha

    def forward(self, x_t, x_f):  # x_t:(B, T_t, D), x_f:(B, T_f, D)
        if not self.enable:
            return x_t, x_f

        q_t = self.q_t(x_t)
        kv_f = self.kv_f(x_f)
        q_f = self.q_f(x_f)
        kv_t = self.kv_t(x_t)

        # time attends to freq  (query=q_t, key/value=kv_f)
        t2, _ = self.mha_t2f(q_t, kv_f, kv_f, need_weights=False)
        # freq attends to time
        f2, _ = self.mha_f2t(q_f, kv_t, kv_t, need_weights=False)

        x_t = self.ln_t(x_t + self.alpha * t2)
        x_t = x_t + self.ffn_t(x_t)
        x_f = self.ln_f(x_f + self.alpha * f2)
        x_f = x_f + self.ffn_f(x_f)
        return x_t, x_f

# ----- The drop-in model -----
# 模型概览：复用原 SSVEPformerX 频域主干，可选接入时域支路并在 CrossView 前融合
class SSVEPformerX_CVFusion(nn.Module):
    """
    兼容原 SSVEPformerX 的频域输入，并在需要时启用时域预览支路。
    - 默认行为保持纯频域；
    - 当提供时域输入时，在 CrossView 前沿通道维拼接后一次性融合。
    """
    def __init__(self,
                 depth_freq: int,
                 depth_time: int,
                 attention_kernal_length: int,
                 chs_num: int,
                 class_num: int,
                 token_dim: int,
                 dropout: float = 0.1,
                 fusion_mode: str = "chan_cat",
                 use_len_fusion_block: bool = True,
                 use_subband: bool = True,
                 subband_proj: str = "linear",
                 use_crossview: bool = True,
                 crossview_alpha: float = 1.0,
                 use_harmonic_pe: bool = True,
                 harmonic_beta: float = 1.0,
                 pe_mode: str = "harmonic",
                 enable_time_branch: bool = True,
                 time_token_dim: int = 250,
                 resize_time_to_freq: bool = True):
        super().__init__()
        self.chs_num = chs_num
        self.class_num = class_num
        self.depth_time = depth_time
        self.depth_freq = depth_freq
        self.attention_kernal_length = attention_kernal_length
        self.token_num = chs_num * 2
        self.token_dim = token_dim
        self.dropout = dropout
        self.use_subband = use_subband
        self.subband_proj = subband_proj
        self.use_crossview = use_crossview
        self.crossview_alpha = crossview_alpha
        self.use_harmonic_pe = use_harmonic_pe
        self.harmonic_beta = harmonic_beta
        self.pe_mode = pe_mode
        self.enable_time_branch = enable_time_branch
        self.time_token_dim = time_token_dim
        self.fusion_mode = fusion_mode
        # 频域补丁嵌入：1x1 卷积把 C 通道映射到 2C token；示例输入 torch.Size([4, 8, 560]) -> 输出 torch.Size([4, 16, 560])
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(chs_num, self.token_num, kernel_size=1, padding=0, groups=1),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 频域谐波位置编码：为每个 token 注入谐波先验；输出与输入保持同形状
        self.harmonic_pe = HarmonicPE(
            token_num=self.token_num,
            token_dim=self.token_dim,
            enable=use_harmonic_pe,
            beta=self.harmonic_beta,
            mode=self.pe_mode
        )
        # 子带门控：多尺度 depthwise 卷积筛选频段；示例输入 torch.Size([4, 16, 560]) -> 输出 torch.Size([4, 16, 560])
        self.subband = (SubBandGate(self.token_num, self.token_dim, k_list=(5, 9, 17),
                                    proj_type=self.subband_proj,
                                    in_ch=self.token_num,
                                    dropout=dropout)
                        if self.use_subband else nn.Identity())

        # CrossViewBlock：视图交叉注意力；示例输入 torch.Size([4, 16, 560]) -> 输出 torch.Size([4, 16, 560])
        self.crossview = (CrossViewBlock(self.token_dim, num_heads=4,
                                         alpha=self.crossview_alpha,
                                         dropout=dropout)
                          if self.use_crossview else nn.Identity())

        # 频域主干：轻量 TinyTransformer；示例输入 torch.Size([4, 16, 560]) -> 输出 torch.Size([4, 16, 560])
        self.backbone = TinyTransformer(depth=self.depth_freq,
                                        token_num=self.token_num,
                                        token_dim=self.token_dim,
                                        ksize=attention_kernal_length,
                                        dropout=dropout)


        # 时域支路：结构与频域对称，方便双支路对齐
        if self.enable_time_branch:
            # 时域补丁嵌入：示例输入 torch.Size([4, 8, 250]) -> 输出 torch.Size([4, 16, 250])
            self.to_patch_embedding_time = nn.Sequential(
                nn.Conv1d(self.chs_num, self.token_num, kernel_size=1, padding=0, bias=False),
                nn.LayerNorm(time_token_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )

            # 时域 TinyTransformer：示例输入 torch.Size([4, 16, 250]) -> 输出 torch.Size([4, 16, 250])
            self.backbone_time = TinyTransformer(
                depth=self.depth_time,
                token_num=self.token_num,
                token_dim=time_token_dim,
                ksize=self.attention_kernal_length,
                dropout=self.dropout
            )

            # 时域谐波位置编码：与频域保持一致的设计
            self.harmonic_pe_time = HarmonicPE(
                token_num=self.token_num,
                token_dim=time_token_dim,
                enable=self.use_harmonic_pe,
                beta=self.harmonic_beta,
                mode=self.pe_mode
            )
        else:
            self.to_patch_embedding_time = None
            self.backbone_time = None
            self.harmonic_pe_time = None

        # 时域长度对齐：可选线性插值把时域长度调到 freq token_dim；示例输出 torch.Size([4, 16, 560])
        self.resize_time_to_freq = resize_time_to_freq and self.enable_time_branch
        if self.resize_time_to_freq and time_token_dim != self.token_dim:
            self.time_resizer = lambda x: F.interpolate(x, size=self.token_dim, mode="linear", align_corners=False)
        else:
            self.time_resizer = None

        # 频域分类头：保持与原模型一致；示例输出 torch.Size([B, class_num])
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

        # 双支路分类头：处理拼接后的 4C token；示例输入 torch.Size([B, 32, token_dim]) -> 输出 torch.Size([B, class_num])
        fused_token_num = self.token_num * 2
        self.head_dual = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(self.token_dim * fused_token_num, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, class_num)
        )

        if self.fusion_mode == "len_cat":
            fused_len = self.time_token_dim + self.token_dim  # T + F
            self.len_fusion_backbone = TinyTransformer(
            depth=2,
            token_num=self.token_num,  # 仍然是 2C 个 token
            token_dim=fused_len,  # 长度维变成 T+F
            ksize=self.attention_kernal_length,
            dropout=self.dropout
            ) if use_len_fusion_block else None

            hidden = class_num * 6
            self.head_len = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(self.dropout),
                nn.Linear(fused_len * self.token_num, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(hidden, class_num)
            )
        else:
            #旧方案不需要，但为避免属性缺失，置 None
            self.len_fusion_backbone = None
            self.head_len = None

        # 参数初始化：沿用原模型，卷积/线性层采用 N(0, 0.01)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_freq, x_time=None):
        """
        x_freq: [B, C, F]  频域输入（预处理生成的复谱特征）
        x_time: [B, C, T]  时域输入（可选；None 时退回单支路）
        """
        # --- freq branch ---
        if x_freq.size(1) != self.chs_num or x_freq.size(-1) != self.token_dim:
            raise ValueError(
                f"SSVEPformerX_CVFusion expects freq input (B,{self.chs_num},{self.token_dim}), "
                f"but got {tuple(x_freq.shape)}"
            )

        xf = self.to_patch_embedding(x_freq)  # (B, 2C, F)
        xf = self.subband(xf)  # (B, 2C, F)
        xf = self.backbone(xf)  # (B, 2C, F)

        use_time_branch = self.enable_time_branch and (x_time is not None)

        # --- single-branch (freq only) ---
        if not use_time_branch:
            # 维持你现在的行为：单路直接走 head（不经 crossview）
            out = self.head(xf)  # (B, num_classes)
            return out

        # --- time branch ---
        if x_time.size(1) != self.chs_num:
            raise ValueError(
                f"Time branch expects channel dim {self.chs_num}, but got {tuple(x_time.shape)}"
            )

        xt = self.to_patch_embedding_time(x_time)  # (B, 2C, T)
        pet = self.harmonic_pe_time(xt)
        xt = xt + pet
        xt = self.backbone_time(xt)  # (B, 2C, T)

        # 对齐时/频长度
        if self.fusion_mode == "len_cat":
            # A 方案：长度维拼接（不插值，保留 T 的细节）
            x_len_cat = torch.cat([xt, xf], dim=2)  # (B, 2C, T+F)
            if self.len_fusion_backbone is not None:
                x_len_cat = self.len_fusion_backbone(x_len_cat)  # (B, 2C, T+F)
            return self.head_len(x_len_cat)  # (B, num_classes)

        elif self.fusion_mode == "chan_cat":
            # 结构化双流：不再把时/频混在一起；必要时仅把时域长度对齐到频域，**分别**送入 CrossView
            if self.time_resizer is not None and xt.size(-1) != xf.size(-1):
                xt = self.time_resizer(xt)  # (B, 2C, F)

    # CrossView 期望 shape (B, L, D)，把 (B, 2C, F) 视作 L=2C, D=F
            x_t = xt                      # (B, 2C, F)
            x_f = xf                      # (B, 2C, F)
            x_t, x_f = self.crossview(x_t, x_f)   # 双流交互
            x_dual = torch.cat([x_t, x_f], dim=1)
            return self.head_dual(x_dual)       # 修改 head_dual 接口（见下）
        else:
            raise ValueError("Unknown fusion_mode=... choose 'len_cat' or 'chan_cat'.")


__all__ = ["SSVEPformerX_CVFusion"]


if __name__ == "__main__":
    # Quick validation mirroring MTSNet to ensure shape compatibility.
    batch_size, chs, time_len, freq_len, num_classes = 4, 8, 250, 560, 40
    freq_input = torch.randn(batch_size, chs, freq_len)
    time_input = torch.randn(batch_size, chs, time_len)

    model = SSVEPformerX_CVFusion(
        depth_freq=1,depth_time=2, attention_kernal_length=31, chs_num=chs, class_num=num_classes,
        token_dim=freq_len, dropout=0.5,
        enable_time_branch=True, time_token_dim=time_len,
        resize_time_to_freq=True,  # 对 len_cat 没影响，可留
        fusion_mode="len_cat",  # ← A 方案
        use_len_fusion_block=True
    )
    print("A) len_cat:", model(freq_input, time_input).shape)

    # chan_cat（旧方案）——插值到 F，通道轴拼接+CrossView
    model = SSVEPformerX_CVFusion(
        depth_freq=1,depth_time=2, attention_kernal_length=31, chs_num=chs, class_num=num_classes,
        token_dim=freq_len, dropout=0.5,
        enable_time_branch=True, time_token_dim=time_len,
        resize_time_to_freq=True,
        fusion_mode="chan_cat"  # ← 旧方案
    )
    print("B) chan_cat:", model(freq_input, time_input).shape)
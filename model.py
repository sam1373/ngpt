# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) The official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except ImportError:
    flash_attn_func = None

"""
def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)

    # Prepare rotations for q and k
    # Interleave real and imaginary parts: For a vector [q0,q1,q2,q3,...],
    # rotate pairs (q_even, q_odd) by [cos, sin].
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)

    # Reshape and apply cos/sin rotations
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1] // 2, 2))
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1] // 2, 2))

    q_rot = q_rot * torch.stack((cos, sin), dim=-1)
    k_rot = k_rot * torch.stack((cos, sin), dim=-1)

    # Reshape back to original dimensions
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)

    return q_rot, k_rot
"""

def apply_rotary_position_embeddings(sinusoidal_pos, q):

    # assume right-aligned
    if sinusoidal_pos.shape[0] != q.shape[2]:
        # print(f"angles shape {angles.shape} is not equal to t shape {t.shape}")
        sinusoidal_pos = sinusoidal_pos[-q.shape[2]:]

    sin, cos = sinusoidal_pos.chunk(2, dim=-1)

    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)

    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1] // 2, 2))

    q_rot = q_rot * torch.stack((cos, sin), dim=-1)

    q_rot = torch.reshape(q_rot, q.shape)

    return q_rot


def get_sinusoidal_embeddings(n_positions, dim):
    """
    Generate sinusoidal positional embeddings.

    Args:
        n_positions (int): Number of positions.
        dim (int): Embedding dimensionality.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (n_positions, dim).
    """
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


class Block(nn.Module):
    """
    A single Transformer block that includes:
    - Multi-head attention (possibly flash attention).
    - An MLP feed-forward layer.
    - Rotary position embeddings.
    - Optional RMSNorm and nGPT scaling layers.
    - Support for custom attention modifications (local heads, thresholding).

    This block can be configured to work as standard GPT or nGPT variants.
    """

    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        # Linear layers for attention Q, K, V, and output projection
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        # MLP layers
        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        self.iblock = iblock
        self.n_head = self.config.n_head

        # Softmax-like configuration for thresholding
        if self.config.softmax_like in ['soft_score_threshold', 'score_threshold']:
            self.thr_c = nn.Parameter(1.6 * torch.ones(self.config.n_head, dtype=torch.float32),
                                      requires_grad=self.config.learned_threshold)
            self.stp = nn.Parameter(10.0 * torch.ones(self.config.n_head, dtype=torch.float32), requires_grad=False)


        # Normalization and scaling depending on use_nGPT
        if self.config.use_nGPT == 0:
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)
        else:
            # nGPT scaling parameters
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(
                self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(
                self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(
                self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(
                self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32)
            )

    def justnorm(self, x):
        """
        Normalize tensor x along the last dimension using L2 norm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: L2-normalized tensor.
        """
        return x / x.norm(p=2, dim=-1, keepdim=True)

    def forward(self, h, past_key_value=None, use_cache=False):
        """
        Forward pass of the Transformer block.

        Args:
            h (torch.Tensor): Input tensor of shape (B, T, C).
            past_key_value (tuple, optional): A tuple of (past_k, past_v) for cached attention states.
            use_cache (bool): If True, returns present_key_value for caching.

        Returns:
            (torch.Tensor, tuple or None): Output tensor and optionally the present_key_value for caching.
        """
        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if self.config.use_nGPT == 0:
            softmax_scale = 1.0 / sqrt_head_dim
        else:
            softmax_scale = sqrt_head_dim

        B, T, C = h.size()

        # Attention RMSNorm (if not using nGPT)
        hin = self.rmsnorm_att(h) if (self.config.use_nGPT == 0) else h

        # Compute Q, K, V
        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)


        total_length = T
        if past_key_value is not None:
            past_k, past_v = past_key_value
            total_length = past_k.shape[1] + T


        if past_key_value is not None:
            k0, v0 = past_key_value
            k = torch.cat([k0, k], dim=1)
            v = torch.cat([v0, v], dim=1)

        if use_cache:
            past_key_value = (k, v)

        # Compute sinusoidal embeddings for rotary position embeddings
        pos = get_sinusoidal_embeddings(total_length, self.config.n_embd // self.config.n_head).to(device=q.device)

        use_self_extend = self.config.self_extend and T > self.config.block_size

        if use_self_extend:
            denominator = max((self.config.block_size - self.config.self_extend_window_size), 1)
            g_size = ((k.shape[2] - self.config.self_extend_window_size + denominator - 1) //
                      denominator * self.config.self_extend_group_multiplier)
            g_size = max(g_size, 1)

            w_size = self.config.self_extend_window_size

            g_pos = pos // g_size
            shift = w_size - (w_size // g_size)
            s_g_pos = g_pos + shift

            q_g = apply_rotary_position_embeddings(s_g_pos, q.transpose(1, 2))
            k_g = apply_rotary_position_embeddings(s_g_pos, k.transpose(1, 2))

            q_g = q_g.to(dtype=torch.bfloat16)
            k_g = k_g.to(dtype=torch.bfloat16)
        else:
            q_g, k_g = None, None

        # Apply rotary position embeddings
        #q, k = apply_rotary_position_embeddings(pos, q.transpose(1, 2), k.transpose(1, 2))
        q = apply_rotary_position_embeddings(pos, q.transpose(1, 2))
        k = apply_rotary_position_embeddings(pos, k.transpose(1, 2))

        #q *= 1.3

        if self.config.use_nGPT == 1:
            # Apply nGPT scaling on Q and K
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
                1, self.config.n_head, 1, self.config.n_embd // self.config.n_head
            )
            q = sqk * self.justnorm(q)
            k = sqk * self.justnorm(k)

            if use_self_extend:
                q_g = sqk * self.justnorm(q_g)
                k_g = sqk * self.justnorm(k_g)

        #q *= 1.3
        #if q_g is not None:
        #    q_g *= 1.3

        # Perform attention
        y = self.attn_maybe_extended(q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16),
                                     v.to(dtype=torch.bfloat16), softmax_scale=softmax_scale,
                                     q_g=q_g, k_g=k_g)
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, C)

        h_att = self.att_c_proj(y)

        # Residual connection or nGPT-style mixing
        if self.config.use_nGPT == 0:
            h = h + h_att
        else:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_att)
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        # MLP layer
        hin = self.rmsnorm_mlp(h) if (self.config.use_nGPT == 0) else h
        uv = self.c_fc(hin)

        if self.config.use_nGPT == 1:
            suv = (self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd ** 0.5)))
            uv = suv * uv

        u, v_ = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v_)
        h_mlp = self.mlp_c_proj(x_mlp)

        # Residual connection or nGPT-style mixing
        if self.config.use_nGPT == 0:
            h = h + h_mlp
        else:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)
            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_mlp)
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        return h, past_key_value

    def attn_maybe_extended(self, q, k, v, softmax_scale, q_g=None, k_g=None):
        """
        Handle attention computation in chunks if attn_chunked is enabled.

        Args:
            q, k, v (torch.Tensor): Q, K, V tensors of shape (B, T, n_head, head_dim).
            softmax_scale (float): The softmax scaling factor.
            q_g, k_g (torch.Tensor, optional): Extended Q, K for self-extend.

        Returns:
            torch.Tensor: The attention output.
        """

        if self.config.attn_chunked:
            B, T, nH, d = q.shape
            chunk_size = self.config.attn_chunk_size
            outs = []
            for start in range(0, T, chunk_size):
                end = start + chunk_size
                q_chunk = q[:, start:end, :, :]
                k_chunk = k[:, :end, :, :]
                v_chunk = v[:, :end, :, :]
                if q_g is not None:
                    q_g_chunk = q_g[:, start:end, :, :]
                    k_g_chunk = k_g[:, :end, :, :]
                else:
                    q_g_chunk, k_g_chunk = None
                out_chunk = self.attn_func(q_chunk, k_chunk, v_chunk, softmax_scale=softmax_scale, q_g=q_g_chunk, k_g=k_g_chunk,
                                           se_w_size=self.config.self_extend_window_size)
                outs.append(out_chunk)
            return torch.cat(outs, dim=1)
        else:
            return self.attn_func(q, k, v, softmax_scale=softmax_scale, q_g=q_g, k_g=k_g,
                                  se_w_size=self.config.self_extend_window_size)

    def attn_func(self, q, k, v, softmax_scale=1.0, q_g=None, k_g=None, se_w_size=None):
        """
        Compute the attention operation.

        Args:
            q, k, v (torch.Tensor): Q, K, V tensors of shape (B, T, n_head, head_dim).
            softmax_scale (float): Scaling factor for attention scores.
            q_g, k_g (torch.Tensor, optional): Extended Q, K.
            se_w_size (int, optional): Self-extend window size.

        Returns:
            torch.Tensor: Attention output of shape (B, T, n_head, head_dim).
        """
        # Transpose to (B, n_head, T, head_dim) for standard attention computation
        #q = q.transpose(1, 2)
        #k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.config.use_flash_attn and flash_attn_func is not None:
            # Use flash attention if available
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=softmax_scale,
                                   causal=True, window_size=(-1, -1), alibi_slopes=None, deterministic=True)

        # Compute raw attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

        # Apply causal mask and window masks
        attn = self.apply_right_aligned_causal_mask(attn)
        attn = self.apply_window_masks(attn, self.iblock)

        # Optional self-extend
        if self.config.self_extend and q_g is not None and k_g is not None:
            attn_g = torch.matmul(q_g, k_g.transpose(-2, -1)) * softmax_scale
            attn_g = self.apply_right_aligned_causal_mask(attn_g)
            attn = self.apply_right_aligned_se_merge(attn, attn_g, se_w_size)

        # Softmax-like function
        attn = self.softmax_like(attn)

        # Compute attention output
        out = torch.matmul(attn, v).transpose(1, 2)
        return out

    def softmax_like(self, scores):
        """
        Applies a softmax-like function to the scores based on config settings.

        Args:
            scores (torch.Tensor): Attention scores (B, n_head, T_q, T_k).

        Returns:
            torch.Tensor: Softmax-like normalized attention weights.
        """
        if self.config.softmax_like == "score_threshold":
            thr, _ = scores.max(dim=-1)
            thr = thr - self.thr_c[None, :, None]
            thr = thr.unsqueeze(-1)
            scores = torch.where(scores < thr, float("-inf"), scores)
            return F.softmax(scores, dim=-1)
        elif self.config.softmax_like == "soft_score_threshold":
            thr, _ = scores.max(dim=-1)
            thr = thr - self.thr_c[None, :, None]
            m = torch.sigmoid((scores - thr.unsqueeze(-1)) * self.stp[None, :, None, None])
            scores = scores + torch.log(m + 1e-8)
            return F.softmax(scores, dim=-1)
        else:
            return F.softmax(scores, dim=-1)

    def apply_right_aligned_causal_mask(self, attn_scores):
        """
        Apply a right-aligned causal mask to the attention scores.

        Args:
            attn_scores (torch.Tensor): Scores of shape (B, n_head, T_q, T_k).

        Returns:
            torch.Tensor: Masked attention scores.
        """
        T_q = attn_scores.size(-2)
        T_k = attn_scores.size(-1)
        device = attn_scores.device
        i = torch.arange(T_q, device=device).unsqueeze(-1)
        j = torch.arange(T_k, device=device).unsqueeze(0)
        causal_mask = (i + (T_k - T_q)) >= j
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        return attn_scores

    def apply_window_masks(self, attn_scores, iblock):
        """
        Apply window masks for local heads during training.

        Args:
            attn_scores (torch.Tensor): (B, n_head, T_q, T_k) attention scores.
            iblock (int): Index of the current block.

        Returns:
            torch.Tensor: Masked attention scores.
        """
        device = attn_scores.device

        # If local heads are used during training
        if self.training and self.config.local_heads_during_training > 0:
            if self.config.local_heads_random:
                head_indices = torch.randperm(self.config.n_head)[:self.config.local_heads_during_training]
            else:
                head_indices = torch.arange(self.config.local_heads_during_training, device=device)

            local_heads_mask = torch.zeros(self.config.n_head, device=device, dtype=torch.bool)
            local_heads_mask[head_indices] = True

            T_q = attn_scores.size(-2)
            T_k = attn_scores.size(-1)
            i = torch.arange(T_q, device=device).view(-1, 1)
            j = torch.arange(T_k, device=device).view(1, -1)
            local_mask = (i - j >= self.config.aug_window_size).bool()
            local_mask = local_mask.unsqueeze(0).unsqueeze(0)

            local_heads_mask_expanded = local_heads_mask.view(1, self.config.n_head, 1, 1)
            attn_scores = attn_scores.masked_fill(local_heads_mask_expanded & local_mask, float('-inf'))

        return attn_scores

    def apply_right_aligned_se_merge(self, attn, attn_g, w_size):
        """
        Merge extended attention 'attn_g' into 'attn' for self-extend functionality.

        Args:
            attn (torch.Tensor): Base attention scores (B, n_head, T_q, T_k).
            attn_g (torch.Tensor): Extended attention scores (B, n_head, T_q, T_k).
            w_size (int): Window size for merging.

        Returns:
            torch.Tensor: Merged attention scores.
        """
        device = attn.device
        T_q = attn.size(-2)
        T_k = attn.size(-1)

        t_start = T_k - T_q
        t_end = T_k

        i = torch.arange(t_start, t_end, device=device).unsqueeze(1)
        j = torch.arange(t_end, device=device).unsqueeze(0)

        cond1 = i >= w_size
        cond2 = (i - w_size) >= j
        mask_chunk = ~(cond1 & cond2)

        # Merge attn and attn_g based on mask_chunk
        attn = torch.where(mask_chunk.unsqueeze(0).unsqueeze(0), attn, attn_g)
        return attn


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model.

    Attributes:
        block_size (int): Sequence length that the model can handle.
        vocab_size (int): Size of the vocabulary.
        n_layer (int): Number of Transformer blocks.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        base_scale (float): Base initialization scaling factor.
        use_nGPT (int): Flag to indicate using nGPT scaling layers.
        dropout (float): Dropout rate.
        bias (bool): Whether to use bias in linear layers.
        use_flash_attn (bool): Whether to use flash attention if available.
        attn_chunked (bool): Whether to process attention in chunks.
        attn_chunk_size (int): Chunk size for attention if attn_chunked is True.
        self_extend (bool): Whether to use self-extend mechanism.
        self_extend_window_size (int): Window size for self-extend merging.
        self_extend_group_multiplier (int): Multiplier for grouping in self-extend.
        softmax_like (str): Type of softmax-like operation to use.
        learned_threshold (bool): Whether threshold parameters are learnable.
        local_heads_during_training (int): Number of local heads to use during training.
        local_heads_random (bool): Whether to randomly choose local heads.
        aug_window_size (int): Window size for augmentation masks.
    """
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False

    use_flash_attn: bool = False
    attn_chunked: bool = False
    attn_chunk_size: int = 256

    self_extend: bool = False
    self_extend_window_size: int = 128
    self_extend_group_multiplier: int = 32

    softmax_like: str = "softmax"
    learned_threshold: bool = False

    local_heads_during_training: int = 0
    local_heads_random: bool = False
    aug_window_size: int = 128


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS norm.

    Attributes:
        weight (nn.Parameter): Scaling parameter.
        eps (float): Epsilon for numerical stability.
    """

    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., embdim).

        Returns:
            torch.Tensor: Normalized tensor of same shape as x.
        """
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):
    """
    GPT Language Model class that ties together:
    - An embedding layer.
    - A stack of Transformer blocks.
    - A final language modeling head.

    Also supports caching past key values for fast generation.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, il) for il in range(config.n_layer)])
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        if config.use_nGPT == 1:
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32))

        if config.use_nGPT == 0:
            self.rmsnorm_f = RMSNorm(config.n_embd)

    def get_num_params(self, non_embedding=True):
        """
        Returns the number of parameters.

        Args:
            non_embedding (bool): If True, returns non-embedding parameters.

        Returns:
            int: Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        """
        Initialize weights of linear and embedding layers.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        """
        Forward pass of the GPT model.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).
            targets (torch.Tensor, optional): Target indices for language modeling. (B, T)
            past_key_values (list of tuples, optional): Past key/value states for caching.
            use_cache (bool): Whether to return present_key_values.

        Returns:
            tuple: (logits, loss, present_key_values if use_cache else None)
        """
        device = idx.device
        # Embedding
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        x = tok_emb
        present_key_values_out = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if (past_key_values is not None) else None
            x, pkv = block(x, past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                present_key_values_out.append(pkv)

        if self.config.use_nGPT == 0:
            x = self.rmsnorm_f(x)

        # Final language modeling head
        if targets is not None:
            logits = self.lm_head(x)
            if self.config.use_nGPT == 1:
                sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
                logits = sz * logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # only last token for inference
            if self.config.use_nGPT == 1:
                sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
                logits = sz * logits
            loss = None

        if use_cache:
            return logits, loss, present_key_values_out
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the optimizer (AdamW).

        Args:
            weight_decay (float): Weight decay parameter.
            learning_rate (float): Learning rate.
            betas (tuple): Betas for AdamW.
            device_type (str): Type of device.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, decode=None):
        """
        Generate text from the model given an initial prompt.

        Args:
            idx (torch.Tensor): Input token indices (B, T).
            max_new_tokens (int): How many new tokens to generate.
            temperature (float): Temperature for sampling.
            top_k (int, optional): If provided, use top-k filtering.

        Returns:
            torch.Tensor: Generated token indices of shape (B, T+max_new_tokens).
        """
        self.eval()
        with torch.no_grad():
            # Initial pass with the entire prompt
            logits, _, past_key_values = self(idx, use_cache=True)

            probs = F.softmax(logits[-1], dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)

            if decode is not None:
                print("predicted:", decode(next_token[0].tolist()))

            for _ in range(max_new_tokens - 1):
                # Generate one token at a time
                logits, _, past_key_values = self(idx[:, -1:], past_key_values=past_key_values, use_cache=True)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    out = torch.full_like(logits, float('-inf'))
                    out.scatter_(1, indices, values)
                    logits = out

                probs = F.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)

                if decode is not None:
                    print("predicted:", decode(next_token[0].tolist()))

        return idx
import torch
import torch.nn as nn


class POD_GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(POD_GRU, self).__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, output_dim)  # Output layer to match input dimensions

    def forward(self, x): # N, grid_x, grid_t, 2
        x = x.permute(0, 2, 1, 3) # N, grid_t, grid_x, 2
        x = x.reshape(x.size(0), x.size(1), -1) # N, grid_t, grid_x * 2
        x = self.in_proj(x) # N, grid_t, hidden_dim
        gru_out, _ = self.gru(x) # N, grid_t, hidden_dim
        out = self.out_proj(gru_out) # N, grid_t, grid_x
        out = out.permute(0, 2, 1) # N, grid_x, grid_t
        out = out.reshape(x.size(0), -1, 1)
        return out

class POD_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(POD_LSTM, self).__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, output_dim)  # Output layer to match input dimensions

    def forward(self, x):
        x = x.permute(0, 2, 1, 3) # N, grid_t, grid_x, 2
        x = x.reshape(x.size(0), x.size(1), -1) # N, grid_t, grid_x * 2
        x = self.in_proj(x) # N, grid_t, hidden_dim
        lstm_out, (_, _) = self.lstm(x)
        out = self.out_proj(lstm_out)
        out = out.permute(0, 2, 1) # N, grid_x, grid_t
        out = out.reshape(x.size(0), -1, 1)
        return out

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.config_mamba import MambaConfig

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class POD_Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        input_dim: int,
        output_dim: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Linear(input_dim, d_model)
        self.out_embedding = nn.Linear(d_model, output_dim)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        input_ids = input_ids.permute(0, 2, 1, 3) # N, grid_t, grid_x, 2
        input_ids = input_ids.reshape(input_ids.size(0), input_ids.size(1), -1) # N, grid_t, grid_x * 2

        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        hidden_states = self.out_embedding(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1) # N, grid_x, grid_t
        hidden_states = hidden_states.reshape(hidden_states.size(0), -1, 1)

        return hidden_states


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
from PositionalEncoding import PositionalEncoding
import math
# from torch_cluster import fps
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class GeGELU(nn.Module):
    """https: // paperswithcode.com / method / geglu"""
    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class StandardAttention(nn.Module):
    """Standard scaled dot product attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., causal=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.causal = causal  # simple autogressive attention with upper triangular part being masked zero

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            if self.causal:
                raise Exception('Passing in mask while attention is not causal')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)     # similarity score

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Contains following two types of attention, as discussed in "Choose a Transformer: Fourier or Galerkin"

    Galerkin type attention, with instance normalization on Key and Value
    Fourier type attention, with instance normalization on Query and Key
    """
    def __init__(self,
                 dim,
                 attn_type,                 # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 init_method='orthogonal',    # ['xavier', 'orthogonal']
                 init_gain=None,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_qkv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    if self.attn_type == 'fourier':
                        # for v
                        init_fn(param[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head, :],
                                gain=self.init_gain)
                        param.data[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head,
                        :] += self.diagonal_weight * \
                              torch.diag(torch.ones(
                                  param.size(-1),
                                  dtype=torch.float32))
                    else: # for galerkin
                        # for q
                        init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                        #
                        param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                    torch.diag(torch.ones(
                                                                                        param.size(-1),
                                                                                        dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, pos=None, not_assoc=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        if self.cat_pos:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1) for x in (q, k, v)]

        if not_assoc:
            # this is more efficient when n<<c
            score = torch.matmul(q, k.transpose(-1, -2))
            out = torch.matmul(score, v) * (1./q.shape[2])
        else:
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class POD_GalerkinTransformer(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 dim_hid,
                 depth,
                 heads,
                 dim_head,
                 attn_type,               # ['standard', 'galerkin', 'fourier']
                 mlp_dim, dropout=0., max_len=5000,):
        super().__init__()

        self.dim_hid = dim_hid

        self.input_emb = nn.Linear(dim_in, dim_hid)

        self.decoder = nn.Linear(dim_hid, dim_out)

        self.pos_encoder = PositionalEncoding(dim_hid, dropout, max_len)

        assert attn_type in ['standard', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim_hid, StandardAttention(dim_hid, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim_hid, FeedForward(dim_hid, mlp_dim, dropout=dropout))
                ]))
        else:

            for _ in range(depth):
                if attn_type == 'galerkin':
                    attn_module = LinearAttention(dim_hid, attn_type, heads=heads, dim_head=dim_head, dropout=dropout)
                else:           # attn_type == 'fourier'
                    attn_module = LinearAttention(dim_hid, attn_type, heads=heads, dim_head=dim_head, dropout=dropout)

                attn_module._init_params()

                self.layers.append(nn.ModuleList([
                    PreNorm(dim_hid, attn_module),
                    FeedForward(dim_hid, mlp_dim, dropout=dropout)
                ]))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3) # N, grid_t, grid_x, 2
        x = x.reshape(x.size(0), x.size(1), -1) # N, grid_t, grid_x * 2

        x = x.permute(1, 0, 2)
        x = self.input_emb(x) * math.sqrt(self.dim_hid)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = self.decoder(x)

        x = x.permute(0, 2, 1) # N, grid_x, grid_t
        x = x.reshape(x.size(0), -1, 1)
        return x




class POD_Transformer(nn.Transformer):
    def __init__(self, ninput, noutput, nhidden, nhead, dim_feedforward, nlayers, dropout=0.0, max_len=5000):
        super(POD_Transformer, self).__init__(d_model=nhidden, nhead=nhead, dim_feedforward=dim_feedforward, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden, dropout, max_len)

        self.input_emb = nn.Linear(ninput, nhidden)
        self.nhidden = nhidden
        self.decoder = nn.Linear(nhidden, noutput)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.bias)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        src = src.permute(0, 2, 1, 3) # N, grid_t, grid_x, 2
        src = src.reshape(src.size(0), src.size(1), -1) # N, grid_t, grid_x * 2
        
        src = src.permute(1, 0, 2)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.nhidden)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)

        output = output.permute(1, 0, 2)

        output = output.permute(0, 2, 1) # N, grid_x, grid_t
        output = output.reshape(output.size(0), -1, 1)
        return output


class MoECrossAttentionBlock(nn.Module):
    def __init__(self, dim_hid, heads, dim_head, n_experts, resid_pdrop=0):
        super(MoECrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim_hid)
        self.ln2 = nn.LayerNorm(dim_hid)
        self.ln3 = nn.LayerNorm(dim_hid)
        self.ln4 = nn.LayerNorm(dim_hid)
        self.ln5 = nn.LayerNorm(dim_hid)

        self.selfattn = LinearAttention(dim_hid, 'galerkin', heads, dim_head)
        self.crossattn = LinearAttention(dim_hid, 'galerkin', heads, dim_head)

        # if config.attn_type == 'linear':
        #     # print('Using Linear Attention')
        #     self.selfattn = LinearAttention(config)
        #     self.crossattn = LinearAttention(config)
        # else:
        #     raise NotImplementedError
        
        self.act = nn.GELU

        # if config.act == 'gelu':
        #     self.act = GELU
        # elif config.act == "tanh":
        #     self.act = Tanh
        # elif config.act == 'relu':
        #     self.act = ReLU
        # elif config.act == 'sigmoid':
        #     self.act = Sigmoid

        self.resid_drop1 = nn.Dropout(resid_pdrop)
        self.resid_drop2 = nn.Dropout(resid_pdrop)

        self.n_experts = n_experts

        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(dim_hid, dim_hid),
            self.act(),
            nn.Linear(dim_hid, dim_hid),
        ) for _ in range(self.n_experts)])

        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(dim_hid, dim_hid),
            self.act(),
            nn.Linear(dim_hid, dim_hid),
        ) for _ in range(self.n_experts)])

        self.gatenet = nn.Sequential(
            nn.Linear(dim_hid, dim_hid),
            self.act(),
            nn.Linear(dim_hid, dim_hid),
            self.act(),
            nn.Linear(dim_hid, self.n_experts)
        )

    '''
        x: [B, T1, C], y:[B, T2, C], pos:[B, T1, n]
    '''
    def forward(self, x):
        gate_score = F.softmax(self.gatenet(x),dim=-1).unsqueeze(2)    # B, T1, 1, m
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln2(x)))
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe1 = (gate_score*x_moe1).sum(dim=-1,keepdim=False)
        x = x + self.ln3(x_moe1)
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        x_moe2 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe2 = (gate_score*x_moe2).sum(dim=-1,keepdim=False)
        x = x + self.ln5(x_moe2)
        return x

class POD_GNOT(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hid, depth, heads, dim_head, n_experts, dropout=0., max_len=5000):
        super(POD_GNOT, self).__init__()

        self.dim_hid = dim_hid

        self.input_emb = nn.Linear(dim_in, dim_hid)

        self.decoder = nn.Linear(dim_hid, dim_out)

        self.pos_encoder = PositionalEncoding(dim_hid, dropout, max_len)

        # self.horiz_fourier_dim = horiz_fourier_dim
        # self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim>0 else trunk_size
        # self.branch_size = branch_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim > 0 else branch_size
        # self.output_size = output_size
        # self.space_dim = space_dim
        # self.gpt_config = MoEGPTConfig(attn_type=attn_type,embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout, attn_pdrop=attn_dropout,n_embd=n_hidden, n_head=n_head, n_layer=n_layers, block_size=128,act=act, n_experts=n_experts,space_dim=space_dim)

        # self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)
        # self.branch_mlp = MLP(self.branch_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)


        self.blocks = nn.Sequential(*[MoECrossAttentionBlock(dim_hid, heads, dim_head, n_experts) for _ in range(depth)])

    def forward(self, x):
        x = x.permute(0, 2, 1, 3) # N, grid_t, grid_x, 2
        x = x.reshape(x.size(0), x.size(1), -1) # N, grid_t, grid_x * 2

        x = x.permute(1, 0, 2)
        x = self.input_emb(x) * math.sqrt(self.dim_hid)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        for block in self.blocks:
            x = block(x)

        x = self.decoder(x)

        x = x.permute(0, 2, 1) # N, grid_x, grid_t
        x = x.reshape(x.size(0), -1, 1)

        return x

if __name__ == "__main__":
    pass
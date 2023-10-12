import polars as pl
import numpy as np

import math
from typing import List, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass(kw_only=True)
class Cox_Config:
    vocab_size:       int
    code_hidden_size: int
    code_num_layers:  int
    code_num_heads:   int
    code_head_dim:    int
    code_intermediate_size:  int
    visit_hidden_size: int
    visit_num_layers:  int
    visit_num_heads:   int
    visit_head_dim:    int
    visit_intermediate_size: int
    dropout_prob: float
    pos_base:     float
    device:       Union[str, torch.device]

SUPER_SMALL_NUMBER = torch.finfo(torch.float).min

class Cox_Model(nn.Module):
    def __init__(self, config: Cox_Config):
        super().__init__()
        self.config = config
        with torch.device(config.device):
            self.embedding = nn.Embedding(config.vocab_size, config.code_hidden_size, padding_idx=0)
            rotary_embedding = Rotary_Embedding (
                config.visit_head_dim,
                config.pos_base,
                device = config.device,
            )
            code_transformer_config = Cox_Transformer_Config (
                hidden_size       = config.code_hidden_size,
                intermediate_size = config.code_intermediate_size,
                head_dim          = config.code_head_dim,
                num_heads         = config.code_num_heads,
                dropout_prob      = config.dropout_prob,
            )
            visit_transformer_config = Cox_Transformer_Config (
                hidden_size       = config.visit_hidden_size,
                intermediate_size = config.visit_intermediate_size,
                head_dim          = config.visit_head_dim,
                num_heads         = config.visit_num_heads,
                dropout_prob      = config.dropout_prob,
            )

            self.code_layers  = nn.ModuleList (
                [
                    Cox_Transformer_Layer(code_transformer_config, None)
                    for _ in range(config.code_num_layers)
                ]
            )
            self.visit_layers = nn.ModuleList (
                [
                    Cox_Transformer_Layer(visit_transformer_config, rotary_embedding)
                    for _ in range(config.visit_num_layers)
                ]
            )
            self.code_to_visit = nn.Linear(config.code_hidden_size, config.visit_hidden_size, bias=False)
            self.head = nn.Linear(config.visit_hidden_size, config.vocab_size, bias=True)

    @profile
    def forward(
        self,
        codes:     torch.LongTensor, # (n_total_visits, n_codes)
        codes_len: torch.LongTensor, # (n_total_visits,)
        visit_id:  torch.LongTensor, # (n_total_visits,)
    ):
        with torch.device(codes.device):
            # code_batch: (n_total_visits, n_codes, code_hidden_size)
            code_batch = self.embedding(codes)
            n_total_visits, n_codes, _ = code_batch.shape

            ### prepare the attention mask for the visit layer transformer

            # attention mask is only a pad mask
            # attention mask has dim (n_total_visits, 1, 1, n_codes)

            filter = torch.arange(n_codes).view((1, -1))
            mask   = torch.full((n_total_visits, n_codes), SUPER_SMALL_NUMBER)
            mask   = mask.masked_fill_(filter < codes_len.view((-1, 1)), 0)
            pad_mask = mask[:, None, None, :]

            ### code level transformer

            for layer in self.code_layers:
                code_batch = layer(code_batch, pad_mask)

            # visits: (n_total_visits, n_codes, visit_hidden_size)
            visits = self.code_to_visit(code_batch)

            # visits: (n_total_visits, visit_hidden_size)
            visits.masked_fill_(mask[..., None] < -1, 0)
            visits = visits.mean(dim=1)

            ### batch data on a visit level rather than a code level

            bsz = visit_id.max() + 1
            patients_len = torch.zeros((bsz,), dtype=int)
            filler = torch.ones((1,), dtype=int).expand(visit_id.shape)
            patients_len.index_add_(0, visit_id, filler)
            batch_dim = patients_len.max()

            starting = torch.empty(patients_len.shape, dtype=int)
            starting[0]  = 0
            starting[1:] = patients_len[:-1].cumsum(0)
            pos = torch.arange(visits.shape[0], dtype=int) - starting[visit_id]
            flatten_pos = batch_dim * visit_id + pos

            t = torch.zeros((bsz * batch_dim, self.config.visit_hidden_size))
            visits = t.index_copy_(0, flatten_pos, visits).view(bsz, batch_dim, self.config.visit_hidden_size)

            ### prepare the attention mask for the visit layer transformer

            # attention mask has dim (bsz, 1, batch_dim, batch_dim)
            # decoder mask has dim (batch_dim, batch_dim)
            # pad mask has dim (bsz, batch_dim)

            decoder_mask = torch.full((batch_dim, batch_dim), SUPER_SMALL_NUMBER).triu_(diagonal=1)

            filter   = torch.arange(batch_dim).view((1, -1))
            pad_mask = torch.full((bsz, batch_dim), SUPER_SMALL_NUMBER)
            pad_mask = pad_mask.masked_fill_(filter < patients_len.view((-1, 1)), 0)

            decoder_mask = decoder_mask[None, None, :, :]
            pad_mask     = pad_mask[:, None, None, :]
            mask = pad_mask + decoder_mask

            ### code level transformer

            for layer in self.visit_layers:
                visits = layer(visits, mask)

            ### final head

            predictions = self.head(visits)

            return predictions, patients_len



# See this [article](http://arxiv.org/abs/2104.09864)
class Rotary_Embedding:
    dim:    int
    base:   float

    def __init__(self, dim: int, base: float, *, device):
        if dim % 2 != 0:
            raise ValueError (f'Tried to instanciate a rotary embedding with a odd dimension ({dim})')
        self.dim  = dim
        self.base = base

        self.max_seq_len = 0
        # @rubustness when we call the .to() method on the parent module, this should be moved to
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    def increase_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t     = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        self.cos_buffer = emb.cos()
        self.sin_buffer = emb.sin()

    def __call__(self, batch: torch.Tensor, positions: torch.LongTensor) -> torch.Tensor:
        bsz, num_heads, b_n, head_dim = batch.shape
        if self.max_seq_len < b_n:
            self.increase_cache(max(b_n, 2*self.max_seq_len))
        batch_x = batch[..., : head_dim // 2  ] 
        batch_y = batch[...,   head_dim // 2 :] 
        rotated = torch.cat((-batch_y, batch_x), dim=-1)
        cos_f = self.cos_buffer[positions].unsqueeze(1) # we will broadcast over dim 1
        sin_f = self.sin_buffer[positions].unsqueeze(1) # we will broadcast over dim 1
        return batch*cos_f + rotated*sin_f

@dataclass(kw_only=True)
class Cox_Transformer_Config:
    hidden_size:       int
    intermediate_size: int
    head_dim:          int
    num_heads:         int
    dropout_prob:      float

class Cox_Transformer_Layer(nn.Module):
    def __init__(
        self,
        config:           Cox_Transformer_Config,
        rotary_embedding: Optional[Rotary_Embedding],
    ):
        super().__init__()
        self.attention = Cox_Attention(config, rotary_embedding)
        self.mlp = Cox_MLP(config.hidden_size, config.intermediate_size)
        self.normalization_pre  = nn.LayerNorm(config.hidden_size)
        self.normalization_post = nn.LayerNorm(config.hidden_size)
        self.dropout_prob = config.dropout_prob

    @profile
    def forward(self, batch: torch.Tensor, mask: torch.Tensor, positions: Optional[torch.LongTensor]=None):
        residual = batch
        batch = self.attention(batch, mask, positions)
        batch = F.dropout(batch, p=self.dropout_prob, training=self.training)
        batch = batch + residual
        residual = batch
        batch = self.normalization_pre(batch)
        batch = self.mlp(batch)
        batch = F.dropout(batch, p=self.dropout_prob, training=self.training)
        batch = batch + residual
        batch = self.normalization_post(batch)
        return batch


class Cox_Attention(nn.Module):
    def __init__(self, config: Cox_Transformer_Config, rotary_embedding: Optional[Rotary_Embedding]):
        super().__init__()
        self.head_dim    = config.head_dim
        self.num_heads   = config.num_heads
        self.hidden_size = config.hidden_size
        self.rotary_embedding = rotary_embedding

        self.q_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    @profile
    def forward (
        self,
        hidden_states: torch.Tensor,
        mask:          torch.Tensor,
        positions:     Optional[torch.LongTensor],
    ):
        bsz, b_n, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)

        if positions is not None:
            query_states = self.rotary_embedding(query_states, positions)
            key_states   = self.rotary_embedding(key_states,   positions)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, b_n, b_n):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, b_n, b_n)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = attn_weights + mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output  = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, b_n, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, b_n, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, b_n, self.num_heads * self.head_dim)
        # @todo
        attn_output = F.dropout(attn_output, p=0.4, training=self.training)
        attn_output = self.o_proj(attn_output)

        return attn_output

class Cox_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def compute_loss(
    prediction:   torch.Tensor,
    output:       torch.Tensor,
    patients_len: torch.LongTensor
) -> torch.Tensor:
    with torch.device(prediction.device):
        filler = torch.arange(prediction.shape[1], dtype=int).unsqueeze(0)
    mask = (filler < patients_len.unsqueeze(1)).unsqueeze(2)
    flat_pred = torch.masked_select(prediction, mask).view(-1, prediction.shape[-1])
    flat_out  = torch.masked_select(output    , mask).view(-1, prediction.shape[-1])

    # if reduce='none' the function returns the same shape as input
    cross_entropy = F.binary_cross_entropy_with_logits(flat_pred, flat_out, reduction='none')
    loss = cross_entropy.sum(dim=-1).mean()
    return loss


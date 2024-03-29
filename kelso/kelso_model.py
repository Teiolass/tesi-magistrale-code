import polars as pl
import numpy as np

import tomlkit

import torch
from torch import nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Self
import os

CFG_FILE_NAME   = 'config.toml' # this is the one that is reported in the save dir
MODEL_FILE_NAME = 'model.torch'

__all__ = [
    'Kelso_Filler', 'Kelso_Predictor', 'load_kelso_for_inference', 'prepare_batch_for_inference',
    'Kelso_Config', 'load_kelso_for_generation', 'prepare_batch_for_generation'
]

@dataclass(kw_only=True)
class Kelso_Config:
    vocab_size:  int
    output_size: int
    hidden_size: int
    num_layers:  int
    num_heads:   int
    head_dim:    int
    pos_base:    float
    dropout:     float
    device:      str | torch.device
    mlp_intermediate_size: int
    parametrized_head: bool


class Kelso_Model(nn.Module):
    def __init__(self, config: Kelso_Config, decoder_mask: bool):
        super().__init__()
        self.config = config
        self.decoder_mask = decoder_mask
        with torch.device(config.device):
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            rotary_embedding = Rotary_Embedding (
                config.head_dim,
                config.pos_base,
                device = config.device,
            )
            self.decoder_layers = nn.ModuleList (
                [Kelso_Decoder_Layer(config, rotary_embedding) for _ in range(config.num_layers)]
            ) 
            self.pooling = Kelso_Pooling(config)
            self.head = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward (
        self,
        batch:     torch.Tensor,
        positions: torch.LongTensor,
        lengths:   torch.LongTensor,
        return_attention: bool = False,
    ) -> list[torch.Tensor]:
        # batch, position size is (bsz, b_n)  
        if batch.shape != positions.shape:
            raise ValueError(
                f'position shape should be equal to batch shape. Expected: {batch.shape}, '
                f'received: {positions.shape}'
            )
        # lenghts size is (bsz,)
        if batch.shape[0] != lengths.shape[0] or len(lengths.shape) >  1:
            raise ValueError(
                f'lengths shape should be ({batch.shape[0]},) but it is {lengths.shape}' 
            )

        bsz, b_n = positions.shape

        # PHASE 1: embed the code_ids

        # embedding has dim (bsz, n, hidden_size)
        embeddings = self.embedding(batch)

        # PHASE 2: create mask

        # attention mask has dim (bsz, 1, b_n, b_n)
        # decoder mask has dim (bsz, b_n, b_n)
        # pad mask has dim (bsz, b_n)

        minus_inf = torch.finfo(torch.float).min

        filter   = torch.arange(b_n, device=self.config.device).view((1, -1))
        pad_mask = torch.full((bsz, b_n), minus_inf, device=self.config.device)
        pad_mask = pad_mask.masked_fill_(filter < lengths.view((-1, 1)), 0)

        pad_mask = pad_mask[:, None, None, :]

        if self.decoder_mask:
            decoder_mask = torch.full((bsz, b_n, b_n), 0.0, device=self.config.device)
            decoder_mask = decoder_mask.masked_fill_(positions.unsqueeze(2) < positions.unsqueeze(1), minus_inf)
            decoder_mask = decoder_mask[:, None, :, :]
            mask = pad_mask + decoder_mask
        else:
            mask = pad_mask

        # PHASE 3: transformer

        all_attentions = []

        batch = embeddings
        for layer in self.decoder_layers:
            batch, attention = layer(batch, positions, mask) 
            all_attentions.append(attention)

        batch = F.dropout(batch, p=self.config.dropout, training=self.training)

        if return_attention:
            # this is of shape (n_layers, bsz, num_heads, b_n, b_n)
            all_attentions = torch.stack(all_attentions)
            # this is of shape (bsz, n_layers, num_heads, b_n, b_n)
            return batch, all_attentions.transpose(0, 1)
        else:
            return batch


class Kelso_Filler(nn.Module):
    def __init__(self, config: Kelso_Config):
        super().__init__()
        self.model  = Kelso_Model(config, decoder_mask=False)
        self.config = config
        with torch.device(config.device):
            self.head = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward (
        self,
        batch:     torch.Tensor,
        positions: torch.LongTensor,
        lengths:   torch.LongTensor,
    ) -> list[torch.Tensor]:
        batch = self.model(batch, positions, lengths)
        result = self.head(batch)
        return result

    

class Kelso_Predictor(nn.Module):
    def __init__(self, config: Kelso_Config):
        super().__init__()
        self.config = config
        self.model  = Kelso_Model(config, decoder_mask=True)
        with torch.device(config.device):
            self.pooling = Kelso_Pooling(config)
            self.head    = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward (
        self,
        batch:     torch.Tensor,
        positions: torch.LongTensor,
        lengths:   torch.LongTensor,
        return_attention: bool = False,
    ) -> list[torch.Tensor]:
        bsz = batch.shape[0]
        batch = self.model(batch, positions, lengths, return_attention)

        if return_attention:
            batch, attention = batch

        # this is of shape (bsz, b_m, hidden)
        batch_pooled = self.pooling(batch, positions, lengths)
        output = self.head(batch_pooled)

        result = []
        for it in range(bsz):
            t = output[it][:positions[it].max() + 1]
            result.append(t)
        if return_attention:
            return result, attention
        else:
            return result

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
        # @rubustness when we call the .to() method on the parent module, this should be moved too
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


class Kelso_Decoder_Layer(nn.Module):
    def __init__(self, config: Kelso_Config, rotary_embedding: Rotary_Embedding):
        super().__init__()
        self.attention = Kelso_Attention(config, rotary_embedding)
        self.mlp = Kelso_MLP(config.hidden_size, config.mlp_intermediate_size)
        self.normalization_pre  = nn.LayerNorm(config.hidden_size)
        self.normalization_post = nn.LayerNorm(config.hidden_size)
        self.dropout = config.dropout
    
    def forward(self, batch: torch.Tensor, positions: torch.LongTensor, mask: torch.Tensor):
        residual = batch
        batch, attn = self.attention(batch, mask, positions)
        batch = F.dropout(batch, p=self.dropout, training=self.training)
        batch = batch + residual
        residual = batch
        batch = self.normalization_pre(batch)
        batch = self.mlp(batch)
        batch = F.dropout(batch, p=self.dropout, training=self.training)
        batch = batch + residual
        batch = self.normalization_post(batch)
        return batch, attn


class Kelso_Attention(nn.Module):
    def __init__(self, config: Kelso_Config, rotary_embedding: Rotary_Embedding|None = None):
        super().__init__()
        self.head_dim         = config.head_dim
        self.num_heads        = config.num_heads
        self.hidden_size      = config.hidden_size
        self.rotary_embedding = rotary_embedding

        self.q_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = config.dropout

    def forward (
        self,
        hidden_states: torch.Tensor,
        mask:          torch.Tensor,
        positions:     torch.LongTensor|None = None,
    ):
        bsz, b_n, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_embedding is not None:
            query_states = self.rotary_embedding(query_states, positions)
            key_states   = self.rotary_embedding(key_states,   positions)
        else:
            if positions is not None:
                raise ValueError('positions provided but no rotary embedding is set')

        attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_logits = attn_logits + mask

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output  = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, b_n, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, b_n, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, b_n, self.num_heads * self.head_dim)
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class Kelso_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Kelso_Pooling(nn.Module):
    def __init__(self, config: Kelso_Config):
        super().__init__()
        self.parametrized_head = config.parametrized_head
        if config.parametrized_head:
            self.scorer_attn = Kelso_Attention(config)
            self.scorer_gmlp = Kelso_MLP(config.hidden_size, config.mlp_intermediate_size)

    def forward(self, batch: torch.Tensor, pos: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        bsz, b_n, h = batch.shape
        b_m = pos.max() + 1
        minus_inf = torch.finfo(torch.float).min
        
        # compute pooling scores
        with torch.device(batch.device):
            decoder_mask = torch.full((bsz, b_n, b_n), 0.0)
            decoder_mask = decoder_mask.masked_fill_(pos.unsqueeze(2) != pos.unsqueeze(1), minus_inf)

            filter   = torch.arange(b_n).view((1, -1))
            pad_mask = torch.full((bsz, b_n), minus_inf)
            pad_mask = pad_mask.masked_fill_(filter < lengths.view((-1, 1)), 0)

        decoder_mask = decoder_mask[:, None, :, :]
        pad_mask = pad_mask[:, None, None, :]
        mask = pad_mask + decoder_mask

        if self.parametrized_head:
            pooling_score = self.scorer_attn(batch, mask)
            pooling_score = self.scorer_gmlp(batch)
        else:
            with torch.device(batch.device):
                pooling_score = torch.ones((bsz, b_n, h))

        # visit-wise sums helpers
        ids = pos.clamp(0, None) + torch.arange(bsz, device=pos.device).unsqueeze(1) * b_m
        ids = ids.flatten()

        # Compute visit-wise softmax
        pooling_mask = torch.zeros((bsz, b_n, 1), device=batch.device)
        pooling_mask.masked_fill_((pos == -1).unsqueeze(2), minus_inf)

        pooling_score += pooling_mask
        pooling_score = torch.exp(pooling_score)
        pooling_score = pooling_score.reshape(-1, h)

        sums = torch.zeros((bsz * b_m, h), device=batch.device)
        sums = sums.index_add_(0, ids, pooling_score)
        pooling_probs = pooling_score / sums[ids]

        # now pool
        batch = batch.reshape((-1, h)) * pooling_probs
        result = torch.zeros((bsz * b_m, h), device=batch.device)
        result = result.index_add_(0, ids, batch)
        result = result.reshape((bsz, b_m, h))

        return result


@dataclass
class Inference_Batch:
    codes:     torch.Tensor
    positions: torch.Tensor
    lenghts:   torch.Tensor

    def unpack(self) -> dict:
        return {
            'batch':     self.codes,
            'positions': self.positions,
            'lengths':   self.lenghts,
        }

def prepare_batch_for_inference (
    codes:     list[np.ndarray],
    counts:    list[np.ndarray],
    positions: list[np.ndarray],
    device:    torch.device,
) -> Inference_Batch:
    codes     = [x.astype(np.int64) for x in codes]
    counts    = [x.astype(np.int64) for x in counts]
    positions = [x.astype(np.int64) for x in positions]

    lengths = [len(x) for x in codes]
    b_n = max(lengths)

    b_codes     = np.array([np.pad(x, (0, b_n - len(x)), constant_values=0 ) for x in codes])
    b_positions = np.array([np.pad(x, (0, b_n - len(x)), constant_values=-1) for x in positions])

    with torch.device(device):
        b_codes     = torch.from_numpy(b_codes)    .to(device)
        b_positions = torch.from_numpy(b_positions).to(device)
        b_lengths   = torch.LongTensor(lengths)    .to(device)

    return Inference_Batch (
        codes     = b_codes,
        positions = b_positions,
        lenghts   = b_lengths,
    )

@dataclass
class Generation_Batch:
    codes:     torch.Tensor
    positions: torch.Tensor
    lenghts:   torch.Tensor

    def unpack(self) -> dict:
        return {
            'batch':     self.codes,
            'positions': self.positions,
            'lengths':   self.lenghts,
        }

def prepare_batch_for_generation (
    codes:     list[np.ndarray],
    counts:    list[np.ndarray],
    positions: list[np.ndarray],
    hole_prob: float,
    hole_token_id: int,
    device:    torch.device,
) -> Generation_Batch:
    codes     = [x.astype(np.int64) for x in codes]
    counts    = [x.astype(np.int64) for x in counts]
    positions = [x.astype(np.int64) for x in positions]

    lengths = [len(x) for x in codes]
    b_n = max(lengths)

    b_codes     = np.array([np.pad(x, (0, b_n - len(x)), constant_values=0 ) for x in codes])
    b_positions = np.array([np.pad(x, (0, b_n - len(x)), constant_values=-1) for x in positions])

    mask = np.random.rand(*b_codes.shape) < hole_prob
    b_codes[mask] = hole_token_id

    with torch.device(device):
        b_codes     = torch.from_numpy(b_codes)    .to(device)
        b_positions = torch.from_numpy(b_positions).to(device)
        b_lengths   = torch.LongTensor(lengths)    .to(device)

    return Inference_Batch (
        codes     = b_codes,
        positions = b_positions,
        lenghts   = b_lengths,
    )




def load_kelso_for_inference(path: str) -> Kelso_Predictor:
    config_path = os.path.join(path, CFG_FILE_NAME)
    model_path  = os.path.join(path, MODEL_FILE_NAME)

    with open(config_path, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)['model']
    config = Kelso_Config(**config)

    model = Kelso_Predictor(config) 
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def load_kelso_for_generation(path: str) -> Kelso_Filler:
    config_path = os.path.join(path, CFG_FILE_NAME)
    model_path  = os.path.join(path, MODEL_FILE_NAME)

    with open(config_path, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)
    hole_prob     = config['trainer']['hole_prob']
    hole_token_id = config['trainer']['hole_token_id']
    config = Kelso_Config(**config['model'])

    model = Kelso_Filler(config) 
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    return model, hole_prob, hole_token_id


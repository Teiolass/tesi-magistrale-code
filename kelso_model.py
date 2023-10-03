import polars as pl
import numpy as np

import math

import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass(kw_only=True)
class Kelso_Config:
    vocab_size:  int
    hidden_size: int
    batch_size:  int
    num_layers:  int
    num_heads:   int
    head_dim:    int
    pos_base:    float
    device:      str

# See this [article](http://arxiv.org/abs/2104.09864)
class Rotary_Embedding:
    dim:    int
    base:   float

    def __init__(self, dim: int, base: float, *, device):
        self.dim  = dim
        self.base = base

        self.max_seq_len = 0
        # @rubustness when we call the .to() method on the parent module, this should be moved to
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    def increase_cache(self, seq_len: int):
        # @bug the comment below is false!!
        # sin_buffer and cos_buffer have shape (b_n, hidden_size)
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
        sin_f = self.sin_buffer[positions].unsqueeze(1)
        return batch*cos_f + rotated*sin_f



class Kelso_Model(nn.Module):
    def __init__(self, config: Kelso_Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        rotary_embedding = Rotary_Embedding (
            config.head_dim,
            config.pos_base,
            device = 'cpu', # @bug this is wrong!
        )
        self.decoder_layers = nn.ModuleList([Kelso_Decoder_Layer(config, rotary_embedding) for _ in range(config.num_layers)]) 

    def forward(self, batch: torch.Tensor, positions: torch.LongTensor, lengths: torch.LongTensor):
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
        # decoder mask has dim (b_n, b_n)
        # pad mask has dim (bsz, b_n)
        decoder_mask = torch.full((b_n, b_n), torch.finfo(torch.float).min, device=config.device)
        decoder_mask = torch.triu(decoder_mask, diagonal=1)

        lengths = lengths.view((-1, 1))
        filter   = torch.arange(b_n, device=config.device).view((1, -1))
        pad_mask = torch.full((bsz, b_n), torch.finfo(torch.float).min, device=config.device)
        pad_mask = pad_mask.masked_fill_(filter < lengths, 0)

        decoder_mask = decoder_mask[None, None, :, :]
        pad_mask = pad_mask[:, None, None, :]
        mask = pad_mask + decoder_mask

        # example of mask, corresponds to a sequence of length 6, padded from a sequence of length 3:
        # (1s are actually -inf in implementation)
        # 
        # 0 1 1 1 1 1
        # 0 0 1 1 1 1
        # 0 0 0 1 1 1
        # 0 0 0 1 1 1
        # 0 0 0 1 1 1
        # 0 0 0 1 1 1

        # PHASE 3: transformer

        _ = self.decoder_layers[0](embeddings, positions, mask)

class Kelso_Attention(nn.Module):
    def __init__(self, config: Kelso_Config, rotary_embedding: Rotary_Embedding):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size 
        self.rotary_embedding = rotary_embedding

        self.q_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def forward (
        self,
        hidden_states: torch.Tensor,
        positions:     torch.LongTensor,
        mask:          torch.Tensor,
    ):
        bsz, b_n, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = self.rotary_embedding(query_states, positions)
        key_states   = self.rotary_embedding(key_states,   positions)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, b_n, b_n):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, b_n, b_n)}, but is"
                f" {attn_weights.size()}"
            )

        if mask.size() != (bsz, 1, b_n, b_n):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, b_n, b_n)}, but is {mask.size()}"
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
        attn_output = self.o_proj(attn_output)

        return attn_output


class Kelso_Decoder_Layer(nn.Module):
    def __init__(self, config: Kelso_Config, rotary_embedding: Rotary_Embedding):
        super().__init__()
        self.attention = Kelso_Attention(config, rotary_embedding)

    
    def forward(self, batch: torch.Tensor, positions: torch.LongTensor, mask: torch.Tensor):
        self.attention(batch, positions, mask)


if __name__ == '__main__':
    diagnoses = pl.read_parquet('data/processed/diagnoses.parquet')
    ccs_codes = pl.read_parquet('data/processed/codes.parquet')

    config = Kelso_Config (
        vocab_size  = ccs_codes['code_id'].shape[0],
        hidden_size = 256,
        batch_size  = 8,
        num_layers  = 3,
        num_heads   = 4,
        head_dim    = 128,
        pos_base    = 10_000,
        device      = 'cpu',
    )

    model = Kelso_Model(config)

    batch = diagnoses.head(config.batch_size)

    b_codes     = list(batch['code_id'].to_numpy())
    b_positions = list(batch['position'].to_numpy())
    b_counts    = list(batch['count'].to_numpy())

    lengths = [len(x) for x in b_codes]
    b_n = max(lengths)
    b_codes     = np.array([np.pad(x, (0, b_n - len(x))) for x in b_codes])
    b_positions = np.array([np.pad(x, (0, b_n - len(x))) for x in b_positions]).astype(np.int_)

    b_codes     = torch.from_numpy(b_codes)
    b_positions = torch.from_numpy(b_positions)
    b_lengths  = torch.LongTensor(lengths)


    model(b_codes, b_positions, b_lengths)


    

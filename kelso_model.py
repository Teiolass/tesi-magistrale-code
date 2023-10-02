import polars as pl
import numpy as np

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
    rotary_embedding_base: float

class Kelso_Model(nn.Module):
    def __init__(self, config: Kelso_Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList([Kelso_Decoder_Layer(config) for _ in range(config.num_layers)]) 

    def forward(self, batch: torch.Tensor, positions: torch.LongTensor, mask: torch.Tensor):
        # batch, position size is (bsz, n)  
        # mask size is (bsz, n, n)?
        if batch.shape != positions.shape: raise ValueError(
                f'position size should be equal to batch size. Expected: {batch.shape}, '
                f'received: {positions.shape}'
            )

        # embedding has dim (bsz, n, hidden_size)
        embeddings = self.embedding(batch)

class Kelso_Decoder_Layer(nn.Module):
    def __init__(self, config: Kelso_Config):
        super().__init__()
    
    def forward(self, batch: torch.Tensor, positions: torch.LongTensor, mask: torch.Tensor):
        pass

class Rotary_Embedding:
    def __init__(self, config: Kelso_Config):
        self.dim = config.hidden_size
        self.base = config.rotary_embedding_base
        self.max_seq_len = 0
        # @rubustness when we call the .to() method on the parent module, this should be moved to
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def increase_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t     = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        self.cos_buffer = emb.cos()
        self.sin_buffer = emb.sin()

    def embed(self, batch: torch.Tensor, positions: torch.LongTensor) -> torch.Tensor:
        # batch has dim (bsz, b_n)
        bsz, bn = batch.shape
        if self.max_seq_len < bn:
            self.increase_cache(max(bn, 2*self.max_seq_len))
        batch_x = batch[:, : bn // 2  ] 
        batch_y = batch[:,   bn // 2 :] 
        rotated = torch.cat((-batch_y, batch_x), dim=-1)
        return batch*cos[positions] + rotated*sin[positions]




if __name__ == '__main__':
    diagnoses = pl.read_parquet('data/processed/diagnoses.parquet')
    ccs_codes = pl.read_parquet('data/processed/codes.parquet')

    config = Kelso_Config (
        vocab_size  = ccs_codes['code_id'].shape[0],
        hidden_size = 256,
        batch_size  = 8,
        num_layers  = 3,
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

    mask = np.zeros((b_n, b_n)) # @todo this is not a mask

    b_codes     = torch.from_numpy(b_codes)
    b_positions = torch.from_numpy(b_positions)
    mask        = torch.from_numpy(mask)

    model(b_codes, b_positions, mask)


    

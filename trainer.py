import polars as pl
import numpy as np

import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from dataclasses import dataclass

from kelso_model import Kelso_Config, Kelso_Model, compute_loss

def log(txt: str):
    tqdm.write(txt)

@dataclass(kw_only=True)
class Trainer_Config:
    batch_size:      int
    num_epochs:      int
    learning_rate:   float
    max_patient_len: int # @bug
    limit_num_batches: Optional[int] = None
    eval_split: float
    test_split: float
    weight_decay: float


def train(model: nn.Module, diagnoses: pl.DataFrame, trainer_config: Trainer_Config):
    codes     = list(diagnoses['code_id'] .to_numpy())
    positions = list(diagnoses['position'].to_numpy())
    counts    = list(diagnoses['count']   .to_numpy())

    num_eval = int(trainer_config.eval_split * len(codes))
    num_test = int(trainer_config.test_split * len(codes))
    p = num_eval + num_test
    splits   = ((p, len(codes)), (0, num_eval), (num_eval, p))

    # the order is train, eval, test. However they are taken from the dataset 
    # in the eval, test, train order
    split_fn = lambda x, y: (x[b[0]:b[1]] for b in y)
    codes_train, codes_eval, codes_test = split_fn(codes, splits)
    positions_train, positions_eval, positions_test = split_fn(positions, splits)
    counts_train, counts_eval, counts_test = split_fn(counts, splits)

    # find the number of batches
    num_batches = len(codes_train) // trainer_config.batch_size
    if trainer_config.limit_num_batches is not None:
        l_num_batches = min(num_batches, trainer_config.limit_num_batches)
        log(
            f'dataset would contain {num_batches} batches of {trainer_config.batch_size}'
            f' but we are limiting it to {l_num_batches}'
        )
        num_batches = l_num_batches
    else:
        log(f'dataset contains {num_batches} batches of {trainer_config.batch_size}')

    #configure the optimizer
    optimizer = torch.optim.AdamW (
        model.parameters(),
        trainer_config.learning_rate,
        weight_decay = trainer_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (
        optimizer,
        factor = 0.4,
        patience = 5,
    )

    # train loop

    for epoch in tqdm(range(trainer_config.num_epochs)):
        for batch_id in tqdm(range(num_batches), leave=False):
        # for batch_id in range(num_batches):
            batch_start = batch_id * trainer_config.batch_size
            batch_end   = batch_start + trainer_config.batch_size

            # get the right data for the batch
            i_codes     = codes_train     [batch_start: batch_end]
            i_positions = positions_train [batch_start: batch_end]
            i_counts    = counts_train    [batch_start: batch_end]

            b_codes, b_positions, b_lengths, outputs = prepare_data (
                i_codes, i_positions, i_counts,
            )

            # feed-forward + backpropagation
            optimizer.zero_grad()
            predictions = model(b_codes, b_positions, b_lengths)
            loss = compute_loss(predictions, outputs) / len(predictions)
            loss.backward()
            optimizer.step()


def prepare_data(i_codes, i_positions, i_counts,):
    # prepare the data for the model
    b_codes     = [x[:-int(c[-1])]                 for x, c in zip(i_codes,     i_counts)]
    b_positions = [x[:-int(c[-1])].astype(np.int_) for x, c in zip(i_positions, i_counts)]
    lengths = [len(x) for x in b_codes]
    b_n     = max(lengths)
    b_codes     = np.array([np.pad(x, (0, b_n - len(x)), constant_values=0 ) for x in b_codes])
    b_positions = np.array([np.pad(x, (0, b_n - len(x)), constant_values=-1) for x in b_positions])
    b_codes     = torch.from_numpy(b_codes)    .to(model.config.device)
    b_positions = torch.from_numpy(b_positions).to(model.config.device)
    b_lengths   = torch.LongTensor(lengths)    .to(model.config.device)

    # compute expected outputs for the loss
    outputs = []
    for it in range(b_codes.shape[0]):
        sz = len(i_counts[it])
        out = np.zeros((sz-1, model.config.vocab_size))
        cursor = i_counts[it][0]
        for jt in range(1, sz):
            cursor_end = cursor + i_counts[it][jt]
            out[jt-1, i_codes[it][cursor:cursor_end]] = 1
            cursor = cursor_end
        out = torch.from_numpy(out).to(model.config.device)
        outputs.append(out)

    return b_codes, b_positions, b_lengths, outputs


if __name__ == '__main__':
    diagnoses = pl.read_parquet('data/processed/diagnoses.parquet')
    ccs_codes = pl.read_parquet('data/processed/codes.parquet')

    config = Kelso_Config (
        vocab_size  = ccs_codes['code_id'].shape[0],
        hidden_size = 64,
        num_layers  = 12,
        num_heads   = 8,
        head_dim    = 32,
        pos_base    = 10_000,
        device      = 'cuda',
        mlp_intermediate_size = 256,
    )
    config.device = torch.device(config.device)
    model = Kelso_Model(config)

    num_params  = sum([param.nelement()                      for param in model.parameters()])
    size_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    log(f'model has {num_params} params, occupying {size_params/1024/1024:.2f}M of memory')


    trainer_config = Trainer_Config (
        batch_size        = 64,
        num_epochs        = 4,
        learning_rate     = 1e-5,
        limit_num_batches = None,
        max_patient_len   = 300,
        eval_split        = 0.15,
        test_split        = 0.15,
        weight_decay      = 1e-3,
    )

    diagnoses = diagnoses.filter(pl.col('count').list.sum() < trainer_config.max_patient_len)

    train (
        model          = model,
        diagnoses      = diagnoses,
        trainer_config = trainer_config,
    )

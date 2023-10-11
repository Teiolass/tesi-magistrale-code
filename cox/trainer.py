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
class Metrics_Config:
    recalls: List[int]
    loss:    bool

@dataclass(kw_only=True)
class Trainer_Config:
    batch_size:        int
    num_epochs:        int
    learning_rate:     float
    max_patient_len:   int # @todo
    limit_num_batches: Optional[int] = None
    eval_split:        float
    test_split:        float
    weight_decay:      float
    eval_batch_size:   int

    # auto-filled
    metrics_config: Optional[Metrics_Config] = None

@dataclass(kw_only=True)
class Batch_Data:
    codes:     torch.LongTensor
    visit_id:  torch.LongTensor
    codes_len: torch.LongTensor
    output:    torch.Tensor


def train(model: nn.Module, diagnoses: pl.DataFrame, trainer_config: Trainer_Config):
    codes = list(diagnoses['code_id'] .to_numpy())

    num_eval = int(trainer_config.eval_split * len(codes))
    num_test = int(trainer_config.test_split * len(codes))
    p = num_eval + num_test
    splits   = ((p, len(codes)), (0, num_eval), (num_eval, p))
    # the order is train, eval, test. However they are taken from the dataset 
    # in the eval, test, train order
    codes_train, codes_eval, codes_test = (codes[b[0]:b[1]] for b in splits)

    train_dataset = prepare_data(codes_train, config.batch_size, config.vocab_size)
    eval_dataset  = prepare_data(codes_eval, config.eval_batch_size, config.vocab_size)

    #configure the optimizer
    optimizer = torch.optim.AdamW (
        model.parameters(),
        trainer_config.learning_rate,
        weight_decay = trainer_config.weight_decay,
    )

    # Train Loop

    for epoch in tqdm(train_dataset, 'epoch'):
        total_train_loss   = 0
        train_loss_divisor = 0

        for batch in tqdm(range(num_batches), 'train', leave=False):
            # feed-forward + backpropagation
            optimizer.zero_grad()
            model.train()
            predictions = model(batch.codes, batch.codes_len, batch.visit_id)
            loss        = compute_loss(predictions, batch.outputs)
            loss        = total_loss
            loss.backward()
            optimizer.step()

            total_train_loss   += float(total_loss)
            train_loss_divisor += 1

        train_loss = total_train_loss / train_loss_divisor
        metrics = evaluate (
            eval_dataset,
            trainer_config,
        )

        txt  = f'{epoch: >3}. '
        txt += f'train_loss: {train_loss:.3f}'
        if trainer_config.metrics_config.loss:
            txt += f"   loss: {metrics['loss']:.3f}"
        for k in trainer_config.metrics_config.recalls:
            txt += f"    r{k}: {metrics['recall'][k]*100:.1f}%"
        txt += f"   lr:{optimizer.param_groups[0]['lr']:.3e}"
        log(txt)


def evaluate (
    dataset: List[Batch_Data]
    config: Trainer_Config,
):
    # prepare the metrics dict
    metrics = {}
    if config.metrics_config.loss:
        metrics['loss'] = 0
    metrics['recall'] = {}
    for k in config.metrics_config.recalls:
        metrics['recall'][k] = 0
    divisor = 0

    num_batches = len(codes) // config.eval_batch_size
    for batch_id in tqdm(range(num_batches), 'eval', leave=False):
        batch_start = batch_id * config.eval_batch_size
        batch_end   = batch_start + config.eval_batch_size

        # get the right data for the batch
        i_codes     = codes     [batch_start: batch_end]
        i_positions = positions [batch_start: batch_end]
        i_counts    = counts    [batch_start: batch_end]

        b_codes, b_positions, b_lengths, outputs = prepare_data (
            i_codes, i_positions, i_counts,
        )

        # computations
        model.eval()
        # with torch.no_grad(): # @todo works?
        with torch.inference_mode():
            predictions = model(b_codes, b_positions, b_lengths)
            m = compute_metrics(predictions, outputs, config.metrics_config)
        if config.metrics_config.loss:
            metrics['loss'] += m['loss']
        for k in config.metrics_config.recalls:
            metrics['recall'][k] += m['recall'][k]
        divisor += sum([x.shape[0] for x in predictions])

    if config.metrics_config.loss:
        metrics['loss'] /= divisor
    for k in config.metrics_config.recalls:
        metrics['recall'][k] /= divisor
    return metrics


# @todo complete return type annotation
def compute_metrics (predictions: List[torch.Tensor], outputs: List[torch.Tensor], config: Metrics_Config):
    metrics = {}
    if config.loss:
        metrics['loss'] = compute_loss(predictions, outputs)
    metrics['recall'] = {}
    for k in config.recalls:
        rec = []
        for pred, out in zip(predictions, outputs):
            # create shifter to index the flatten array
            t = torch.ones(out.shape[0], dtype=int, device=out.device) * out.shape[-1]
            t[0] = 0
            t = t.cumsum(0).unsqueeze(1)

            sel = pred.topk(k, dim=-1).indices
            tp = out.flatten()[sel+t].sum(-1).to(float) # true positives
            tt = out.sum(-1)
            recall = (tp / tt).mean()
            rec.append(float(recall))
        metrics['recall'][k] = sum(rec)
    return metrics

def prepare_data(i_codes: List[np.ndarray], batch_size: int, vocab_size: int) -> List[Batch_Data]:
    # @todo 
    pin_memory: bool = True

    i_codes = [list(x) for x in i_codes]
    dataset = []

    cursor = 0
    while True:
        end_cursor = cursor + batch_size
        if end_cursor >= len(i_codes):
            break

        b_codes = i_codes[cursor:end_cursor]

        l_codes     = []
        l_codes_len = []
        l_visit_id  = []

        for it, patient in enumerate(b_codes[:-1]):
            l_codes     += patient
            l_codes_len += [len(x) for x in patient]
            l_visit_id  += [it] * len(patient)

        codes_len = torch.LongTensor(l_codes_len, pin_memory=pin_memory)
        visit_id  = torch.LongTensor(l_visit_id,  pin_memory=pin_memory)

        max_code_len = max(l_codes_len)
        padded_codes = [x + [0] * (max_code_len - len(x)) for x in l_codes]

        codes = torch.LongTensor(padded_codes, pin_memory=pin_memory)

        max_visit_len = max(l_visit_id) + 1
        output = torch.zeros((len(b_codes), max_visit_len, vocab_size), pin_memory=pin_memory)

        for it, patient in enumerate(b_codes[1:]):
            for jt, codes in enumerate(patient):
                for c in codes:
                    output[it, jt, c] = 1

        batch_data = Batch_Data (
            codes       = codes,
            visit_id    = visit_id,
            codes_len   = codes_len,
            output      = output,
        )

        cursor = end_cursor
        dataset.append(batch_data)
    return dataset



class Scheduler:
    def __init__(self, optimizer, min_lr, max_lr, warmup_steps, gamma):
        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer     = optimizer
        self.min_lr        = min_lr
        self.max_lr        = max_lr
        self.warmup_steps  = warmup_steps
        self.gamma         = gamma
        self.epoch_counter = 0

    def step(self):
        self.epoch_counter += 1
        if self.epoch_counter < self.warmup_steps:
            lr = self.min_lr + (self.max_lr - self.min_lr) * self.epoch_counter / self.warmup_steps
        else:
            lr = max(self.max_lr * self.gamma ** (self.epoch_counter - self.warmup_steps), self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



if __name__ == '__main__':
    diagnoses = pl.read_parquet('data/processed/diagnoses.parquet')
    ccs_codes = pl.read_parquet('data/processed/codes.parquet')

    train (
        model          = model,
        diagnoses      = diagnoses,
        trainer_config = trainer_config,
    )

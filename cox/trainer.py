import polars as pl
import numpy as np

import math
from typing import List, Optional, Self

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import tomlkit

from dataclasses import dataclass

from cox_model import Cox_Config, Cox_Model, compute_loss

def log(txt: str):
    tqdm.write(txt)

@dataclass(kw_only=True)
class Metrics_Config:
    recalls: List[int]

@dataclass(kw_only=True)
class Trainer_Config:
    batch_size:        int
    num_epochs:        int
    learning_rate:     float
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

    def to(self, device: torch.device) -> Self:
        return Batch_Data (
            codes     = self.codes.to(device),
            visit_id  = self.visit_id.to(device),
            codes_len = self.codes_len.to(device),
            output    = self.output.to(device),
        )


def train(model: nn.Module, diagnoses: pl.DataFrame, config: Trainer_Config):
    codes = list(diagnoses['code_id'] .to_numpy())

    num_eval = int(trainer_config.eval_split * len(codes))
    num_test = int(trainer_config.test_split * len(codes))
    p = num_eval + num_test
    splits = ((p, len(codes)), (0, num_eval), (num_eval, p))
    # the order is train, eval, test. However they are taken from the dataset 
    # in the eval, test, train order
    codes_train, codes_eval, codes_test = (codes[b[0]:b[1]] for b in splits)

    train_dataset = prepare_data(codes_train, config.batch_size, model.config.vocab_size)
    eval_dataset  = prepare_data(codes_eval, config.eval_batch_size, model.config.vocab_size)

    #configure the optimizer
    optimizer = torch.optim.AdamW (
        model.parameters(),
        trainer_config.learning_rate,
        weight_decay = trainer_config.weight_decay,
    )

    # Train Loop

    for epoch in tqdm(range(config.num_epochs), 'epoch'):
        total_train_loss = 0

        for batch in tqdm(train_dataset, 'train', leave=False):
            batch = batch.to(model.config.device)

            # feed-forward + backpropagation
            optimizer.zero_grad()
            model.train()
            predictions, patients_len = model(batch.codes, batch.codes_len, batch.visit_id)
            loss = compute_loss(predictions, batch.output, patients_len)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss)

        train_loss = total_train_loss / len(train_dataset)
        metrics = evaluate (
            eval_dataset,
            trainer_config,
        )

        txt  = f'{epoch: >3}. '
        txt += f'train_loss: {train_loss:.3f}'
        txt += f"   loss: {metrics['loss']:.3f}"
        for k in trainer_config.metrics_config.recalls:
            txt += f"    r{k}: {metrics['recall'][k]*100:.1f}%"
        txt += f"   lr:{optimizer.param_groups[0]['lr']:.3e}"
        log(txt)


def evaluate (
    dataset: List[Batch_Data],
    config: Trainer_Config,
) -> dict:
    # prepare the metrics dict
    metrics = {}
    metrics['loss'] = 0
    metrics['recall'] = {}
    for k in config.metrics_config.recalls:
        metrics['recall'][k] = 0

    for batch in tqdm(dataset, 'eval', leave=False):
        batch = batch.to(model.config.device)

        # computations
        model.eval()
        # with torch.no_grad(): # @todo works?
        with torch.inference_mode():
            prediction, patients_len = model(batch.codes, batch.codes_len, batch.visit_id)
            m = compute_metrics(prediction, batch.output, patients_len, config.metrics_config)

        metrics['loss'] += m['loss']
        for k in config.metrics_config.recalls:
            metrics['recall'][k] += m['recall'][k]

    metrics['loss'] /= len(dataset)
    for k in config.metrics_config.recalls:
        metrics['recall'][k] /= len(dataset)
    return metrics


def compute_metrics (
    prediction:   torch.Tensor,
    output:       torch.Tensor,
    patients_len: torch.LongTensor,
    config:       Metrics_Config,
) -> dict:
    metrics = {}
    # @todo this should be inlined: we don't need to reevaluate the flat_pred and flat_out
    metrics['loss'] = compute_loss(prediction, output, patients_len)
    metrics['recall'] = {}

    with torch.device(prediction.device):
        filler = torch.arange(prediction.shape[1], dtype=int).unsqueeze(0)
        mask = (filler < patients_len.unsqueeze(1)).unsqueeze(2)
        flat_pred = torch.masked_select(prediction, mask).view(-1, prediction.shape[-1])
        flat_out  = torch.masked_select(output    , mask).view(-1, prediction.shape[-1])

        for k in config.recalls:
            best = flat_pred.topk(k, dim=-1).indices
            offsets = flat_out.stride(-1) * torch.arange(best.size(0))
            indices = best + offsets.unsqueeze(1)
            sel = (flat_out.flatten()[indices.flatten()]).view(best.shape)

            nums = sel.sum(dim=-1)
            dens = flat_out.sum(dim=-1)
            recs = nums / dens
            rec = recs.mean()

            metrics['recall'][k] = float(rec)
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

        max_visit_len = 0
        for it, patient in enumerate(b_codes):
            l_codes     += patient[:-1]
            l_codes_len += [len(x) for x in patient[:-1]]
            l_visit_id  += [it] * len(patient[:-1])
            max_visit_len = max(max_visit_len, len(patient)-1)

        codes_len = torch.tensor(l_codes_len, dtype=int, pin_memory=pin_memory)
        visit_id  = torch.tensor(l_visit_id,  dtype=int, pin_memory=pin_memory)

        max_code_len = max(l_codes_len)
        padded_codes = [list(x) + [0] * (max_code_len - len(x)) for x in l_codes]

        codes = torch.tensor(padded_codes, dtype=int, pin_memory=pin_memory)

        output = torch.zeros((len(b_codes), max_visit_len, vocab_size), pin_memory=pin_memory)

        for it, patient in enumerate(b_codes):
            for jt, p_codes in enumerate(patient[1:]):
                for c in p_codes:
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



# @todo what about this??
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
    config_file_path = 'rem/cox/config.toml'

    log(f'Using config file: {config_file_path}')
    with open(config_file_path, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)

    diagnoses = pl.read_parquet(config['diagnoses_path'])
    ccs_codes = pl.read_parquet(config['ccs_codes_path'])

    config['model']['vocab_size'] = ccs_codes.shape[0]
    model_config = Cox_Config(**config['model'])
    trainer_config = Trainer_Config(**config['train'])
    metrics_config = Metrics_Config(**config['metrics'])

    trainer_config.metrics_config = metrics_config
    model_config.device = torch.device(model_config.device)

    model = Cox_Model(model_config)

    train (
        model     = model,
        diagnoses = diagnoses,
        config    = trainer_config,
    )

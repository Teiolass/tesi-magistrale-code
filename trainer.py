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
    codes_train,     codes_eval,     codes_test     = split_fn(codes,     splits)
    positions_train, positions_eval, positions_test = split_fn(positions, splits)
    counts_train,    counts_eval,    counts_test    = split_fn(counts,    splits)

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
    # @debug
    # scheduler = Scheduler (
    #     optimizer    = optimizer,
    #     min_lr       = 1e-5,
    #     max_lr       = 1e-4,
    #     warmup_steps = 15,
    #     gamma        = 0.995,
    # )

    # @debug
    # check the initial conditions
    # metrics = evaluate (
    #     codes_eval,
    #     positions_eval,
    #     counts_eval,
    #     trainer_config,
    # )
    # txt = 'Initial metrics:  '
    # if trainer_config.metrics_config.loss:
    #     txt += f"   loss: {metrics['loss']:.3f}"
    # for k in trainer_config.metrics_config.recalls:
    #     txt += f"    r{k}: {metrics['recall'][k]*100:.1f}%"
    # txt += f"   lr:{optimizer.param_groups[0]['lr']:.3e}"
    # log(txt)

    # Train Loop

    for epoch in tqdm(range(trainer_config.num_epochs), 'epoch'):
        total_train_loss   = 0
        train_loss_divisor = 0

        for batch_id in tqdm(range(num_batches), 'train', leave=False):
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
            model.train()
            predictions = model(b_codes, b_positions, b_lengths)
            total_loss  = compute_loss(predictions, outputs)
            divisor     = sum([x.shape[0] for x in predictions])
            loss        = total_loss / divisor
            loss.backward()

            ##############   @debug
            show_batch_grad = []
            if batch_id in show_batch_grad:
                log(f'======== BATCH ID {batch_id} ========')
                xx = {}
                for i, layer in enumerate(model.decoder_layers):
                    for n, p in layer.named_parameters():
                        v = p.abs().mean()
                        if n not in xx:
                            xx[n] = [v]
                        else:
                            xx[n].append(v)
                for key, v in xx.items():
                    txt = f'{key: <30}:  '
                    for x in v:
                        txt += f'{x: <8.1e} '
                    log(txt)
                log(f'head weigths: {model.head.weight.grad.abs().mean():.1e}')
                log(f'head bias   : {model.head.bias  .grad.abs().mean():.1e}')
                log('')
            ##############

            optimizer.step()

            total_train_loss   += float(total_loss)
            train_loss_divisor += divisor

        train_loss = total_train_loss / train_loss_divisor
        metrics = evaluate (
            codes_eval,
            positions_eval,
            counts_eval,
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

        # scheduler.step()



def evaluate (
    codes:     List[np.ndarray],
    positions: List[np.ndarray],
    counts:    List[np.ndarray],
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


# @todo type annotations
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
        out = np.zeros((sz-1, model.config.vocab_size), dtype=float)
        cursor = i_counts[it][0]
        for jt in range(1, sz):
            cursor_end = cursor + i_counts[it][jt]
            out[jt-1, i_codes[it][cursor:cursor_end]] = 1
            cursor = cursor_end
        out = torch.from_numpy(out).to(torch.float).to(model.config.device)
        outputs.append(out)
    return b_codes, b_positions, b_lengths, outputs

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

    config = Kelso_Config (
        vocab_size  = ccs_codes['code_id'].shape[0],
        hidden_size = 256,
        num_layers  = 0,
        num_heads   = 16,
        head_dim    = 64,
        pos_base    = 8_000,
        device      = 'cuda',
        mlp_intermediate_size = 2048,
    )
    config.device = torch.device(config.device)
    model = Kelso_Model(config)

    num_params  = sum([param.nelement()                      for param in model.parameters()])
    size_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    log(f'model has {num_params/1e6:.2f}M params, occupying {size_params/1024/1024:.2f}M of memory')


    trainer_config = Trainer_Config (
        batch_size        = 64,
        num_epochs        = 1000,
        learning_rate     = 3e-4,
        limit_num_batches = None,
        max_patient_len   = 150,
        eval_split        = 0.20,
        test_split        = 0.15,
        weight_decay      = 0.2,
        eval_batch_size   = 128,
    )
    metrics_config = Metrics_Config (
        recalls = [5, 10, 20, 30],
        loss    = True,
    )

    trainer_config.metrics_config = metrics_config

    num_patients_before = diagnoses.shape[0]
    diagnoses = diagnoses.filter(pl.col('count').list.sum() < trainer_config.max_patient_len)
    num_patients_after = diagnoses.shape[0]
    log (
        f'Original dataset contains {num_patients_before} patients. We cut '
        f'those with more than {trainer_config.max_patient_len} codes, so we are '
        f'left with {num_patients_after} patients ' 
        f'({(1 - num_patients_after / num_patients_before)*100:.1f}% less).'
    )

    train (
        model          = model,
        diagnoses      = diagnoses,
        trainer_config = trainer_config,
    )

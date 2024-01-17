import math
import os
from dataclasses import dataclass

import polars as pl
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import tomlkit
from datetime import datetime

from kelso_model import Kelso_Config, Kelso_Predictor, CFG_FILE_NAME, MODEL_FILE_NAME

CONFIG_FILE_PATH = 'repo/kelso/hyper_config.toml'

LOG_FILE_NAME = 'log.txt'
CSV_FILE_NAME = 'log.csv'

def nice_print(txt: str):
    tqdm.write(txt)

@dataclass(kw_only=True)
class Metrics_Config:
    recalls: list[int]

@dataclass(kw_only=True)
class Trainer_Config:
    batch_size:        int
    num_epochs:        int
    learning_rate:     float
    eval_split:        float
    test_split:        float
    weight_decay:      float
    eval_batch_size:   int
    ccs_as_inputs:     bool
    save_directory:    str
    patience:          int
    max_patient_len:   int # @todo
    limit_num_batches: int | None = None

    # auto-filled
    metrics_config: Metrics_Config | None = None

@dataclass(kw_only=True)
class Epoch_Metrics:
    epoch:      int
    learn_rate: float
    train_loss: float
    eval_loss:  float
    recalls:    dict[int, float]

def log_metrics(metrics: Epoch_Metrics, trainer_config: Trainer_Config):
    txt_file_path = os.path.join(trainer_config.save_directory, LOG_FILE_NAME)
    csv_file_path = os.path.join(trainer_config.save_directory, CSV_FILE_NAME)

    txt  = f'{metrics.epoch: >3}. '
    txt += f'train_loss: {metrics.train_loss:.3f}'
    txt += f'   eval loss: {metrics.eval_loss:.3f}'
    for k in trainer_config.metrics_config.recalls:
        txt += f"    r{k}: {metrics.recalls[k]*100:.2f}%"
    txt += f"   lr:{metrics.learn_rate:.3e}"

    nice_print(txt)
    with open(txt_file_path, 'a') as f:
        f.write(txt + '\n')

    if metrics.epoch == 0:
        if os.path.exists(csv_file_path):
            raise Error(f'We are in epoch zero, but csv file {csv_file_path} already exists')
        txt  = 'epoch,learn_rate,train_loss,eval_loss,'
        txt += ','.join([f'recall_{k}' for k in sorted(metrics.recalls)]) 
        txt += '\n'
    else:
        txt = ''

    values = [
        metrics.epoch,
        metrics.learn_rate,
        metrics.train_loss,
        metrics.eval_loss,
    ]
    values += [metrics.recalls[k] for k in sorted(metrics.recalls)]
    values  = [str(v) for v in values]
    txt += ','.join(values)
    txt += '\n'

    with open(csv_file_path, 'a') as f:
        f.write(txt)


@dataclass(kw_only=True)
class Early_Stopper_Result:
    is_best_round:  bool
    should_exit: bool

class Early_Stopper:
    best_result: float
    best_epoch:  int | None
    patience:    int
    min_is_better: bool

    def __init__(self, patience: int, *, min_is_better: bool):
        self.best_result = 0.0
        self.best_epoch  = None
        self.patience    = patience
        self.min_is_better = min_is_better

    def check(self, epoch: int, metric: float) -> Early_Stopper_Result:
        check = self.best_epoch is None or metric < self.best_result
        if check == self.min_is_better:
            # then we are doing good
            self.best_epoch  = epoch
            self.best_result = metric
            is_best_round  = True
            should_exit = False
        else:
            is_best_round  = False
            should_exit = epoch - self.best_epoch > self.patience
        return Early_Stopper_Result(is_best_round=is_best_round, should_exit=should_exit)

@dataclass(kw_only=True)
class Training_Results:
    num_epochs: int
    loss: float
    recalls: dict[int, float]

class Complete_Search:
    def __init__(self, values: dict[str, list]):
        self.values = values
        self.current = {key: 0 for key in values.keys()}
        self.is_finished = False

    def next(self) -> dict|None:
        if self.is_finished: return None
        ret = {key: self.values[key][self.current[key]] for key in self.values.keys()}

        is_finished = True
        for key in self.values.keys():
            self.current[key] += 1
            if self.current[key] >= len(self.values[key]):
                self.current[key] = 0
            else:
                is_finished = False
                break
        self.is_finished = is_finished
        return ret

    def __iter__(self):
        return self

    def __next__(self):
        x = self.next()
        if x is None: raise StopIteration
        return x

class Search_Logger:
    params:  list[dict]
    results: list[Training_Results]
    paths:   list[str]

    def __init__(self):
        self.params  = []
        self.results = []
        self.paths   = []

    def log(self, param: dict, result: Training_Results, path: str):
        self.paths  .append(path)
        self.params .append(param)
        self.results.append(result)

    def best_loss(self) -> int:
        index_best = -1
        best = float('inf')
        for it, r in enumerate(self.results):
            if r.loss < best:
                best = r.loss
                index_best = it
        return it

    def save_log(self, save_path: str):
        param_keys = list(self.params[0].keys())
        recalls    = sorted(self.results[0].recalls)
        log_keys   = ['loss'] + ['recall@{}'.format(k) for k in recalls]

        header = ','.join(['path'] + param_keys + log_keys)
        lines = [header]
        for path, param, res in zip(self.paths, self.params, self.results):
            data  = ['"{}"'.format(path)] 
            data += [str(param[key]) for key in param_keys]
            data += [str(res.loss)] + [str(res.recalls[k]) for k in recalls]
            line  = ','.join(data)
            lines.append(line)
        
        txt = '\n'.join(lines)

        with open(save_path, 'w') as f:
            f.write(txt)


def train(model: nn.Module, diagnoses: pl.DataFrame, trainer_config: Trainer_Config) -> Training_Results:
    def split(data):
        ccs_codes = list(data['ccs_id'].to_numpy())
        if trainer_config.ccs_as_inputs:
            input_codes = list(data['ccs_id'].to_numpy())
        else:
            input_codes = list(data['icd9_id'].to_numpy())
        positions = list(data['position'].to_numpy())
        counts    = list(data['count']   .to_numpy())
        return ccs_codes, input_codes, counts, positions

    ccs_train, input_train, counts_train, positions_train = split(diagnoses.filter(pl.col('role') == 'train'))
    ccs_eval,  input_eval,  counts_eval,  positions_eval  = split(diagnoses.filter(pl.col('role') == 'eval'))
    ccs_test,  input_test,  counts_test,  positions_test  = split(diagnoses.filter(pl.col('role') == 'eval'))

    # find the number of batches
    num_batches = len(ccs_train) // trainer_config.batch_size
    if trainer_config.limit_num_batches is not None:
        l_num_batches = min(num_batches, trainer_config.limit_num_batches)
        nice_print(
            f'dataset would contain {num_batches} batches of {trainer_config.batch_size}'
            f' but we are limiting it to {l_num_batches}'
        )
        num_batches = l_num_batches
    else:
        nice_print(f'dataset contains {num_batches} batches of {trainer_config.batch_size}')

    model_save_path = os.path.join(trainer_config.save_directory, MODEL_FILE_NAME)

    #configure the optimizer
    optimizer = torch.optim.AdamW (
        model.parameters(),
        trainer_config.learning_rate,
        weight_decay = trainer_config.weight_decay,
    )

    stopper = Early_Stopper(trainer_config.patience, min_is_better=True) 

    # Train Loop

    try:
        for epoch in tqdm(range(trainer_config.num_epochs), 'epoch', leave=False):
            total_train_loss   = 0
            train_loss_divisor = 0

            for batch_id in tqdm(range(num_batches), 'train', leave=False):
                batch_start = batch_id * trainer_config.batch_size
                batch_end   = batch_start + trainer_config.batch_size

                # get the right data for the batch
                i_ccs       = ccs_train       [batch_start: batch_end]
                i_input     = input_train     [batch_start: batch_end]
                i_positions = positions_train [batch_start: batch_end]
                i_counts    = counts_train    [batch_start: batch_end]

                b_codes, b_positions, b_lengths, outputs = prepare_data (
                    i_ccs, i_input, i_positions, i_counts,
                )

                # feed-forward + backpropagation
                optimizer.zero_grad()
                model.train()
                predictions = model(b_codes, b_positions, b_lengths)
                total_loss  = compute_loss(predictions, outputs)
                divisor     = sum([x.shape[0] for x in predictions])
                loss        = total_loss / divisor
                loss.backward()

                optimizer.step()

                total_train_loss   += float(total_loss)
                train_loss_divisor += divisor

            train_loss = total_train_loss / train_loss_divisor
            metrics_dict = evaluate (
                model,
                ccs_eval,
                input_eval,
                positions_eval,
                counts_eval,
                trainer_config,
            )
            
            metrics = Epoch_Metrics (
                epoch      = epoch,
                learn_rate = float(optimizer.param_groups[0]['lr']),
                train_loss = train_loss,
                eval_loss  = metrics_dict['loss'],
                recalls    = metrics_dict['recalls']
            )
            log_metrics(metrics, trainer_config)

            # The `min_is_better` field is relevant in constructor!!
            stopper_result = stopper.check(epoch, metrics.eval_loss)

            if stopper_result.is_best_round:
                torch.save(model.state_dict(), model_save_path)

            if stopper_result.should_exit:
                nice_print('It seems we are done here...')
                break
    except KeyboardInterrupt:
        nice_print('exiting loop...')


    model.load_state_dict(torch.load(model_save_path))
    metrics_dict = evaluate (
        model,
        ccs_test,
        input_test,
        positions_test,
        counts_test,
        trainer_config,
    )
    training_results = Training_Results (
        num_epochs = epoch,
        loss       = metrics_dict['loss'],
        recalls    = metrics_dict['recalls'],
    )
    return training_results


def evaluate (
    model: nn.Module,
    ccs_codes: list[np.ndarray],
    icd_codes: list[np.ndarray],
    positions: list[np.ndarray],
    counts:    list[np.ndarray],
    config:    Trainer_Config,
):
    # prepare the metrics dict
    metrics = {}
    metrics['loss'] = 0
    metrics['recalls'] = {}
    for k in config.metrics_config.recalls:
        metrics['recalls'][k] = 0
    divisor = 0

    num_batches = len(ccs_codes) // config.eval_batch_size
    for batch_id in tqdm(range(num_batches), ' eval', leave=False):
        batch_start = batch_id * config.eval_batch_size
        batch_end   = batch_start + config.eval_batch_size

        # get the right data for the batch
        i_icd       = icd_codes [batch_start: batch_end]
        i_ccs       = ccs_codes [batch_start: batch_end]
        i_positions = positions [batch_start: batch_end]
        i_counts    = counts    [batch_start: batch_end]

        b_codes, b_positions, b_lengths, outputs = prepare_data (
            i_ccs, i_icd, i_positions, i_counts,
        )

        # computations
        model.eval()
        with torch.inference_mode():
            predictions = model(b_codes, b_positions, b_lengths)
            m = compute_metrics(predictions, outputs, config.metrics_config)
        metrics['loss'] += m['loss']
        for k in config.metrics_config.recalls:
            metrics['recalls'][k] += m['recalls'][k]
        divisor += sum([x.shape[0] for x in predictions])

    metrics['loss'] /= divisor
    for k in config.metrics_config.recalls:
        metrics['recalls'][k] /= num_batches
    return metrics

def compute_loss(predictions, outputs) -> torch.Tensor:
    losses = []
    for pred, out in zip(predictions, outputs):
        # if reduce='none' the function returns the same shape as input
        loss = F.binary_cross_entropy_with_logits(pred, out, reduction='sum')
        losses.append(loss)
    total_loss = sum(losses)
    return total_loss


def compute_metrics (predictions: list[torch.Tensor], outputs: list[torch.Tensor], config: Metrics_Config):
    metrics = {}
    metrics['loss'] = float(compute_loss(predictions, outputs))
    metrics['recalls'] = {}
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
        metrics['recalls'][k] = sum(rec) / len(rec)
    return metrics


def prepare_data(i_ccs, i_input, i_positions, i_counts):
    # prepare the data for the model
    b_input     = [x[:-int(c[-1])]                 for x, c in zip(i_input,     i_counts)]
    b_positions = [x[:-int(c[-1])].astype(np.int_) for x, c in zip(i_positions, i_counts)]
    lengths = [len(x) for x in b_input]
    b_n     = max(lengths)
    b_input     = np.array([np.pad(x, (0, b_n - len(x)), constant_values=0 ) for x in b_input]).astype(np.int64)
    b_positions = np.array([np.pad(x, (0, b_n - len(x)), constant_values=-1) for x in b_positions])
    # @todo this relies on the global model variable
    b_input     = torch.from_numpy(b_input)    .to(model.config.device)
    b_positions = torch.from_numpy(b_positions).to(model.config.device)
    b_lengths   = torch.LongTensor(lengths)    .to(model.config.device)

    # compute expected outputs for the loss
    outputs = []
    for it in range(b_input.shape[0]): # this is range(batch_size)
        sz = len(i_counts[it])
        out = np.zeros((sz-1, model.config.output_size), dtype=float)
        cursor = i_counts[it][0]
        for jt in range(1, sz):
            cursor_end = cursor + i_counts[it][jt]
            out[jt-1, i_ccs[it][cursor:cursor_end]] = 1
            cursor = cursor_end
        out = torch.from_numpy(out).to(torch.float).to(model.config.device)
        outputs.append(out)
    return b_input, b_positions, b_lengths, outputs

DIR_ID_LENGTH = 5
ALL_LETTERS   = 'abcdefghijklmnopqrstuvxywz'
def format_path(path: str) -> str:
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    id = ''.join([random.choice(ALL_LETTERS) for _ in range(DIR_ID_LENGTH)])
    path = path.replace('%(id)',   id)
    path = path.replace('%(date)', date)
    return path


if __name__ == '__main__':
    nice_print(f'Using config file: {CONFIG_FILE_PATH}')
    with open(CONFIG_FILE_PATH, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)

    diagnoses = pl.read_parquet(config['diagnoses_path'])

    searcher = Complete_Search(config['hyper_search'])
    search_logger = Search_Logger()

    search_file = format_path(config['trainer']['search_path']) 
    config['trainer'].remove('search_path')

    total_search_iterations = 1
    for it in searcher.values.values():
        total_search_iterations *= len(it)

    for search_params in tqdm(searcher, 'searcher', leave=False, total=total_search_iterations):

        config['model']['vocab_size']  = diagnoses['icd9_id'].list.max().max() + 1
        config['model']['output_size'] = diagnoses['ccs_id'] .list.max().max() + 1

        config['model']['hidden_size'] = search_params['hidden_size']
        config['model']['num_layers']  = search_params['num_layers']
        config['model']['num_heads']   = search_params['num_heads']
        config['model']['head_dim']    = search_params['head_dim']
        config['model']['mlp_intermediate_size'] = search_params['mlp_intermediate_size']
        config['trainer']['learning_rate'] = search_params['learning_rate']


        kelso_config = Kelso_Config(**config['model'])
        kelso_config.device = torch.device(kelso_config.device)
        model = Kelso_Predictor(kelso_config)

        num_params  = sum([param.nelement()                      for param in model.parameters()])
        size_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        nice_print(f'model has {num_params/1e6:.2f}M params, occupying {size_params/1024/1024:.2f}M of memory')

        trainer_config = Trainer_Config(**config['trainer'])
        metrics_config = Metrics_Config(**config['metrics'])
        trainer_config.metrics_config = metrics_config

        trainer_config.save_directory = format_path(trainer_config.save_directory) 
        nice_print(f'> Save directory is {trainer_config.save_directory}')
        os.makedirs(trainer_config.save_directory)

        num_patients_before = diagnoses.shape[0]
        diagnoses = diagnoses.filter(pl.col('count').list.sum() < trainer_config.max_patient_len)
        # @debug
        # diagnoses = diagnoses.filter(pl.col('count').list.lengths() > 2)
        num_patients_after = diagnoses.shape[0]
        nice_print (
            f'Original dataset contains {num_patients_before} patients. We cut '
            f'those with more than {trainer_config.max_patient_len} codes, so we are '
            f'left with {num_patients_after} patients ' 
            f'({(1 - num_patients_after / num_patients_before)*100:.1f}% less).'
        )

        training_infos = tomlkit.table()
        training_infos['num_patients']   = num_patients_after
        training_infos['num_parameters'] = num_params
        training_infos['hyper_search']   = True
        config['training_infos'] = training_infos

        # update toml document
        new_config_text = tomlkit.dumps(config)
        new_config_path = os.path.join(trainer_config.save_directory, CFG_FILE_NAME)
        with open(new_config_path, 'w') as f:
            f.write(new_config_text)

        starting_time = datetime.now()

        results = train (
            model          = model,
            diagnoses      = diagnoses,
            trainer_config = trainer_config,
        )

        search_logger.log(search_params, results, trainer_config.save_directory)

        # compute training time
        end_time = datetime.now()
        time_delta = end_time - starting_time
        config['training_infos']['training_time'] = str(time_delta)

        # add test data to toml document
        test_results = tomlkit.table()
        test_results['training_epochs'] = results.num_epochs
        test_results['loss'] = results.loss
        for k in sorted(metrics_config.recalls):
            test_results[f'recall_{k}'] = results.recalls[k]
        config['test_results'] = test_results

        # save new results on config file
        new_config_text = tomlkit.dumps(config)
        with open(new_config_path, 'w') as f:
            f.write(new_config_text)

        # print the test results on screen
        txt = [f'test loss: {results.loss:.3f}']
        for k in sorted(metrics_config.recalls):
            txt += [f'recall_{k: <2}: {results.recalls[k]*100:.2f}%']
        txt = '    '.join(txt)
        nice_print(txt)

    search_logger.save_log(search_file)


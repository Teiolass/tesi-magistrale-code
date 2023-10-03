import datetime as dt
import os
from typing import TypeVar, Tuple, List
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import torch
from   torch import nn, optim
import torch.nn.functional as F
from   torch.nn.utils.rnn import *
from   torch.linalg import vecdot

import ray
from ray import tune
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch

from mimic_loader import *

device = 'cuda'

class DraiModel(nn.Module):
    embedding : nn.Module
    recurrent : nn.Module
    output    : nn.Module

    n_codes : int
    
    def __init__(self, *, hidden_size: int, n_codes: int, n_layers: int, dropout: float):
        # some housekeeping
        super(DraiModel, self).__init__()

        # create the layers
        # softmax is not included since the CrossEntropyLoss already applies it
        self.embedding = nn.EmbeddingBag (
            num_embeddings      = n_codes,
            embedding_dim       = hidden_size,
            mode                = 'sum',
            include_last_offset = False,
            padding_idx         = 0
        )
        self.recurrent = nn.GRU (
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = 0 if n_layers == 1 else dropout,
        )
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                in_features  = hidden_size,
                out_features = n_codes,
            ),
        )
        self.n_codes = n_codes

    def forward(self, patients: List[torch.Tensor]) -> List[torch.Tensor]:
        embeddings = [self.embedding(patient) for patient in patients]
        input  = pack_sequence(embeddings)
        packed_out, hidden = self.recurrent(input)
        output = unpack_sequence(packed_out) 
        output = [self.output(t) for t in output]
        return output

@dataclass(kw_only=True)
class TrainableData:
    train_patients: np.ndarray
    valid_patients: np.ndarray
    n_codes: int

def train_model(config, data: TrainableData):
    batch_size    = config['batch_size']
    learning_rate = config['learning_rate']
    l2_penalty    = config['l2_penalty']
    hidden_size   = config['hidden_size']
    n_layers      = config['n_layers']
    dropout       = config['dropout']

    epochs        = 1000
    n_codes        = data.n_codes
    train_patients = data.train_patients
    valid_patients = data.valid_patients

    drai = DraiModel (
        hidden_size = hidden_size,
        n_codes     = n_codes,
        n_layers    = n_layers,
        dropout     = dropout,
    ).to(device)
    drai.eval()


    loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    optimizer     = optim.Adam(drai.parameters(), lr=learning_rate, weight_decay=l2_penalty)

    num_batches = int((len(train_patients)) / batch_size)
    for epoch in range(epochs):
        train_loss = 0
        num_loss   = 0

        drai.train()
        for batch_i in range(num_batches):
            start = batch_i * batch_size
            end   = start + batch_size
            batch = sorted(train_patients[start:end], key = lambda t: len(t), reverse=True)
            batch_in  = [t[:-1] for t in batch]
            batch_out = [t[1: ] for t in batch]

            optimizer.zero_grad()
            predictions = drai(batch_in)

            losses = []
            for pred, out in zip(predictions, batch_out):
                full_probs = pred.repeat_interleave(64, 0)
                full_codes = out.reshape((-1))
                losses.append(loss_function(full_probs, full_codes))
                num_loss += pred.shape[0]
            loss = sum(losses)
            loss.backward()
            train_loss += loss
            optimizer.step()
        drai.eval()

        valid_loss, recalls = evaluate(drai, valid_patients, loss_function)
        train_loss = float(train_loss / num_loss)
        record = {}
        record['train_loss'] = float(train_loss)
        record['valid_loss'] = float(valid_loss)
        for k, v in zip(recall_params, recalls):
            field = f'recall_{k}'
            record[field] = float(v)

        tune.report(**record)
        
def format_seconds(seconds: float) -> str:
    if seconds < 100:
        return f'{seconds:.1f} secs'
    mins = seconds / 60
    return f'{mins:.1f} mins'

EVALUATION_BATCH_SIZE = 50
recall_params = [10, 20, 30]


def evaluate(model: DraiModel, data: List[torch.Tensor], loss_function):
    batches = []
    cursor  = 0
    while cursor < len(data):
        end = cursor + EVALUATION_BATCH_SIZE
        batches.append(data[cursor:end])
        cursor = end
    if cursor < len(data) - 1:
        batches.append(data[cursor:])

    total_loss = 0
    num_loss   = 0
    total_recall = [0] * len(recall_params)

    for batch in batches:
        batch = sorted(batch, key = lambda t: len(t), reverse=True)
        batch_in  = [t[:-1] for t in batch]
        batch_out = [t[1: ] for t in batch]
        with torch.no_grad():
            batch_pred = model(batch_in)
        for pred, out in zip(batch_pred, batch_out):
            full_probs = pred.repeat_interleave(64, 0)
            full_codes = out.reshape((-1))
            total_loss += loss_function(full_probs, full_codes)
            for i, k in enumerate(recall_params):
                topk = pred.topk(k, dim=1).indices
                top_pred    = F.one_hot(topk, model.n_codes)[:,:,1:].sum(1).type(torch.HalfTensor)
                one_hot_out = F.one_hot(out , model.n_codes)[:,:,1:].sum(1).type(torch.HalfTensor)
                recalls       = vecdot(top_pred, one_hot_out)
                normalization = one_hot_out.sum(1)
                total_recall[i] += torch.div(recalls, normalization).sum()
            num_loss += pred.shape[0]
    recall_val = [v / num_loss for v in total_recall]
    return total_loss / num_loss, recall_val
    

if __name__ == '__main__':
    # these paths are relevant for the cnr machine only
    if not 'mimic' in globals() or not 'data' in globals():
        print('it seems that the variables `mimic` and `data` are not defined in the global namespace')
        print('I`m going to create them')
        mimic = Mimic.from_folder('/home/amarchetti/data/mimic-iii', '/home/amarchetti/data')
        data  = MimicData.from_mimic(mimic, pad_visits=False, pad_codes=True)
        # data.codes in a [n,64] shaped numpy array, where n is the total number of visits
        #
        # For each i in range(n), data[i,:] is a vector of non-zero diagnoses codes padded with zero codes

        # data.patients is an [m] shaped numpy array of codes, where m-1 is the number of patients
        #
        # data.patients[0] is always 0
        #
        # for each i in range(m-1), the visits in data.codes at indexes in
        # range(data.patients[i], data.patients[i+1]) are the visits corrseponding to
        # the i-th patient
        print('data loaded')
    else:
        print('I have found the variables `mimic` and `data` in the global namespace')
        print('I think there is no need to recompute them')
    print()

    train_ratio   = .7
    patients = []
    for i in range(len(data.patients) - 1):
        start = data.patients[i]
        end   = data.patients[i+1]
        codes_per_patient = data.codes[start:end]
        tensor = torch.tensor(codes_per_patient, dtype=torch.int64, device=device)
        patients.append(tensor)
    num_train = int(len(patients) * train_ratio)
    num_valid = len(patients) - num_train
    train_patients = patients[:num_train]
    valid_patients = patients[num_train:]
    trainable_data = TrainableData (
        train_patients = train_patients,
        valid_patients = valid_patients,
        n_codes        = data.get_num_codes()
    )

    space = {
        'batch_size'    : hp.choice('batch_size', [1,2,4,8,16,24,32,64]),
        'learning_rate' : hp.loguniform('learning_rate', -10, 0),
        'l2_penalty'    : hp.loguniform('l2_penalty', -4, 2),
        'hidden_size'   : hp.choice('hidden_size', [2**k for k in range(5, 12)]),
        'n_layers'      : hp.choice('n_layers', [1,2,3,4,5]),
        'dropout'       : hp.uniform('dropout', 0, 1),
    }

    initial_parameters = [{
        'batch_size'    : 24,
        'learning_rate' : 5e-3,
        'l2_penalty'    : 5e-2,
        'hidden_size'   : 2048,
        'n_layers'      : 2,
        'dropout'       : 0.9,
    }]

    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    scheduler = tune.schedulers.ASHAScheduler (
        max_t            = 800,
        grace_period     = 15,
        reduction_factor = 3,
        metric = 'valid_loss',
        mode   = 'min',
    )
    hyperopt_search = HyperOptSearch (
        space,
        metric = 'valid_loss',
        mode   = 'min',
        points_to_evaluate = initial_parameters,
    )
    tuner = tune.Tuner (
        tune.with_resources(
            tune.with_parameters(train_model, data=trainable_data),
            resources = {'cpu': 1, 'gpu': 1},
        ),
        tune_config = tune.TuneConfig (
            num_samples = 100,
            search_alg  = hyperopt_search,
            scheduler   = scheduler,
        ),
    )
    results = tuner.fit()

    print('\n\n ======== FINAL RESULTS =======\n')
    best_result = results.get_best_result('valid_loss', 'min')
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial train loss: {:.4f}".format(best_result.metrics['train_loss']))
    print("Best trial valid loss: {:.4f}".format(best_result.metrics['valid_loss']))
    print("Best trial recall_10: {:.2f}%".format(best_result.metrics['recall_10'] * 100))
    print("Best trial recall_20: {:.2f}%".format(best_result.metrics['recall_20'] * 100))
    print("Best trial recall_30: {:.2f}%".format(best_result.metrics['recall_30'] * 100))
import datetime as dt
from typing import TypeVar, Tuple, List

import numpy as np

import torch
from   torch import nn, optim
import torch.nn.functional as F
from   torch.nn.utils.rnn import *
from   torch.linalg import vecdot

from mimic_loader import *

device = 'cpu'

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


def train(model, data):
    loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    optimizer     = optim.Adam(model.parameters(), lr=1e-2)

    train_ratio = .7
    batch_size  = 5
    epochs      = 300

    patients = []
    for i in range(len(data.patients) - 1):
        start = data.patients[i]
        end   = data.patients[i+1]
        codes_per_patient = data.codes[start:end]
        tensor = torch.tensor(codes_per_patient, dtype=torch.int64, device=device)
        patients.append(tensor)

    num_train = int(len(patients) * train_ratio)
    num_test  = len(patients) - num_train

    train_patients = patients[:num_train]
    test_patients  = patients[num_train:]

    num_batches = int((len(train_patients)-1) / batch_size)

    print('starting train loop...\n')
    starting_time = dt.datetime.now()

    for epoch in range(epochs):
        train_loss = 0
        num_loss   = 0
        for batch_i in range(num_batches):
            start = batch_i * batch_size
            end   = start + batch_size
            batch = sorted(patients[start:end], key = lambda t: len(t), reverse=True)
            batch_in  = [t[:-1] for t in batch]
            batch_out = [t[1: ] for t in batch]

            optimizer.zero_grad()
            predictions = model(batch_in)

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

        test_loss, recalls = evaluate(model, test_patients, loss_function)
        train_loss = float(train_loss / num_loss)

        now = dt.datetime.now()
        elapsed = (now - starting_time).total_seconds()
        total_time   = format_seconds(elapsed)
        average_time = format_seconds(elapsed / (epoch+1))
        
        print(f'epoch {epoch:<3}. Test loss is {test_loss:<8.3f}. Train loss is {train_loss:<8.3f}')
        print(f'    ', end='')
        for p, r in zip(recall_params, recalls):
            print(f'recall@{p}: {100*r:<4.1f}%    ', end='')
        print('')
        print(f'total time: {total_time:10<} average epoch time {average_time}')
        
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
                one_hot_out = F.one_hot(out, model.n_codes )[:,:,1:].sum(1).type(torch.HalfTensor)
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
        mimic = Mimic.from_folder('/home/amarchetti/mimic-iii', '/home/amarchetti')
        data  = MimicData.from_mimic(mimic, pad_visits=False, pad_codes=True)
        print('data loaded')
    else:
        print('I have found the variables `mimic` and `data` in the global namespace')
        print('I think there is no need to recompute them')
    print()

    n_codes = data.get_num_codes()

    drai = DraiModel (
        hidden_size = 1024,
        n_codes     = n_codes,
        n_layers    = 2,
        dropout     = 0.6,
    ).to(device)

    train(drai, data)
    


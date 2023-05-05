import torch
from   torch import nn, optim
import torch.nn.functional as F
from   torch.linalg import vecdot

import numpy as np

from mimic_loader import *

device = 'cuda'

class Model(nn.Module):
    embedding  : nn.Module
    sequential : nn.Module

    n_codes : int

    def __init__(self, *, hidden_size: int, n_codes: int, n_layers: int, dropout: float):
        super(Model, self).__init__()

        self.embedding = nn.EmbeddingBag (
            num_embeddings      = n_codes,
            embedding_dim       = hidden_size,
            mode                = 'sum',
            include_last_offset = False,
            padding_idx         = 0
        )
        sequential_list = []
        for it in range(n_layers):
            sequential_list.append(
                nn.Linear(
                    in_features  = hidden_size,
                    out_features = hidden_size,
                )
            )
            sequential_list.append(nn.ReLU())
            if dropout > 0:
                sequential_list.append(nn.Dropout(dropout))

        sequential_list.append(
            nn.Linear(
                in_features  = hidden_size,
                out_features = n_codes,
            )
        )
        self.sequential = nn.Sequential(*sequential_list)
        self.n_codes = n_codes

    def forward(self, batch_in):
        codes       = self.embedding(batch_in)
        predictions = self.sequential(codes)
        return predictions

def evaluate(model, batches_in, batches_out, loss_function):
    total_loss = 0
    total_recalls = [0] * len(recall_params)
    num_loss   = 0
    for batch_in, batch_out in zip(batches_in, batches_out):
        with torch.no_grad():
            batch_pred = model(batch_in)
        loss = loss_function(batch_pred.repeat_interleave(64, 0), batch_out.reshape((-1)))
        total_loss += float(loss)
        for i, k in enumerate(recall_params):
            topk = batch_pred.topk(k, dim=1).indices
            top_pred    = F.one_hot(topk     , model.n_codes)[:,:,1:].sum(1).type(torch.HalfTensor)
            one_hot_out = F.one_hot(batch_out, model.n_codes)[:,:,1:].sum(1).type(torch.HalfTensor)
            recall_unnorm = vecdot(top_pred, one_hot_out)
            normalization = one_hot_out.sum(1)
            recall = recall_unnorm = torch.div(recall_unnorm, normalization).sum()
            total_recalls[i] += recall
        num_loss   += len(batch_in)
    average_loss = total_loss / num_loss
    recalls = [v / num_loss for v in total_recalls]
    return average_loss, recalls


def batch(elements, batch_size):
    cursor  = 0
    batches = []
    while cursor < len(elements):
        end = min(cursor + batch_size, len(elements))
        batches.append(elements[cursor:end])
        cursor = end
    return batches


def train(model, data):
    train_ratio   = .7
    batch_size    = 20
    epochs        = 6000
    # patience      = 10
    learning_rate = 2e-4

    loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    optimizer     = optim.Adam(model.parameters(), lr=learning_rate)

    input_list  = []
    output_list = []
    for i in range(len(data.patients) - 1):
        start = data.patients[i]
        end   = data.patients[i+1]
        codes_per_patient = data.codes[start:end]
        tensor = torch.tensor(codes_per_patient, dtype=torch.int64, device=device)
        input_list .append(tensor[:-1])
        output_list.append(tensor[1: ])
    input  = torch.cat(input_list)
    output = torch.cat(output_list)

    num_train = int(input.shape[0] * train_ratio)
    
    train_batches_in  = batch(input [:num_train], batch_size)  
    train_batches_out = batch(output[:num_train], batch_size)

    test_batches_in  = batch(input [num_train:], 100)  
    test_batches_out = batch(output[num_train:], 100)

    global history
    history = []

    for epoch in range(epochs):
        train_total_loss = 0
        for batch_in, batch_out in zip(train_batches_in, train_batches_out):
            optimizer.zero_grad()
            batch_pred = model(batch_in)
            loss = loss_function(batch_pred.repeat_interleave(64, 0), batch_out.reshape((-1)))
            loss.backward()
            optimizer.step()

            train_total_loss += float(loss)

        train_average_loss = train_total_loss / num_train
        test_average_loss, recalls  = evaluate(model, test_batches_in, test_batches_out, loss_function)
        print(f'epoch {epoch:<4} -  Train loss: {train_average_loss:.3f}  -  Test loss: {test_average_loss:.3f}')
        txt = '    '
        for k, v in zip(recall_params, recalls):
            t = f'recall@{k}: {100*v:.2f}%   -   '
            txt += t
        print(txt)

        review = {}
        review['train_average_loss'] = train_average_loss
        review['test_average_loss']  = test_average_loss
        review['recalls']            = recalls
        history.append(review)
    return history

if __name__ == '__main__':
    if not 'mimic' in globals() or not 'data' in globals():
        print('it seems that the variables `mimic` and `data` are not defined in the global namespace')
        print('I`m going to create them')
        mimic = Mimic.from_folder('/home/amarchetti/data/mimic-iii', '/home/amarchetti/data')
        data  = MimicData.from_mimic(mimic, pad_visits=False, pad_codes=True)
        print('data loaded')
    else:
        print('I have found the variables `mimic` and `data` in the global namespace')
        print('I think there is no need to recompute them')
    print()

    n_codes = data.get_num_codes()

    model = Model (
        hidden_size = 2048,
        n_codes     = n_codes,
        n_layers    = 1,
        dropout     = 0.3,
    ).to(device)


    recall_params = [10, 20, 30]
    history = []
    train(model, data)

    


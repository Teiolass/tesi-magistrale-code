import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import Tuple, List


from mimic_loader import *

device = 'cuda'

class LinearModel(nn.Module):
    embedding_bag: nn.Module
    n_embeddings: int

    def __init__(self, *, n_embeddings: int):
        super(LinearModel, self).__init__()
        self.embedding_bag = nn.EmbeddingBag (
            num_embeddings      = n_embeddings,
            embedding_dim       = n_embeddings,
            mode                = 'sum',
            include_last_offset = False,
            padding_idx         = 0,
        )
        self.n_embeddings = n_embeddings

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_bag(codes)
        # predictions = torch.cumsum(embeddings, dim=0)
        return embeddings

def train (
    model     : LinearModel,
    *,
    optimizer : optim.Optimizer,
    data      : MimicData,
    split     : Tuple[int, int],
    epochs    : int,
) -> dict:
    codes = torch.tensor(data.codes, dtype=torch.int64, device=device)
    cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    # hacky way of doing a one hot encoding and then summing all the times
    # is it necessary?
    encoder_weights = torch.eye(model.n_embeddings, device=device)
    encoder_weights[0,0] = 0
    target_encoder = nn.EmbeddingBag (
        num_embeddings      = model.n_embeddings,
        embedding_dim       = model.n_embeddings,
        mode                = 'sum',
        include_last_offset = False,
        device              = device,
        _weight             = encoder_weights,
    )
    num_batches = split[1] - split[0]

    loss_history = []
    for epoch in range(epochs):
        cursor = data.patients[split[0]]
        train_loss = 0
        for p in range(split[0]+1, split[1]):
            optimizer.zero_grad()
            
            new_cursor = data.patients[p]
            prediction_cats = codes[cursor:new_cursor-1]
            predictions = model(prediction_cats)

            targets_cat = codes[cursor+1: new_cursor]
            targets = target_encoder(targets_cat)
            loss = cross_entropy(predictions, targets)
            train_loss += loss

            loss.backward()
            optimizer.step()

            cursor = new_cursor
        average_loss = train_loss/num_batches
        print(f'epoch {epoch} done with loss {average_loss}')

        if len(loss_history) > 0 and average_loss > loss_history[-1]:
            lr = optimizer.param_groups[0]['lr']
            if lr < 1e-7:
                break
            optimizer.param_groups[0]['lr'] = lr * 0.6
        loss_history.append(average_loss)

        if epoch % 4 == 0:
            result = test (
                model,
                data,
                (5000, 6000),
                [10, 20, 30],
            )
            print(result)

    return {
        'loss_history': loss_history,
    }

def test(
    model: LinearModel,
    data: MimicData,
    split: Tuple[int, int],
    recall: List[int],
) -> dict:
    codes = torch.tensor(data.codes, dtype=torch.int64, device=device)
    cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    encoder_weights = torch.eye(model.n_embeddings, device=device)
    encoder_weights[0,0] = 0
    target_encoder = nn.EmbeddingBag (
        num_embeddings      = model.n_embeddings,
        embedding_dim       = model.n_embeddings,
        mode                = 'sum',
        include_last_offset = False,
        device              = device,
        _weight             = encoder_weights,
    )
    num_batches = split[1] - split[0]

    total_loss = 0
    total_recall = [0] * len(recall)
    cursor = data.patients[split[0]]
    for p in range(split[0]+1, split[1]):
        new_cursor = data.patients[p]
        prediction_cats = codes[cursor:new_cursor-1]

        with torch.no_grad():
            predictions = model(prediction_cats)
            targets_cat = codes[cursor+1: new_cursor]
            targets = target_encoder(targets_cat)
            loss = cross_entropy(predictions, targets)
            total_loss += loss

            den = torch.sum(targets, dim=1)
            for i, k in enumerate(recall):
                topk = torch.topk(predictions, k, dim=1).indices
                top_vec = target_encoder(topk)
                tp = torch.bmm(top_vec.unsqueeze(1), targets.unsqueeze(2)).squeeze()
                vals = torch.div(tp, den)
                total_recall[i] += vals.sum() / predictions.shape[0]
        cursor = new_cursor


    return {
        'loss'  : total_loss.tolist() / num_batches,
        'recall': [r.tolist() / num_batches for r in total_recall]
    }


if __name__ == '__main__':
    # these paths are relevant for the cnr machine only
    mimic = Mimic.from_folder('/home/amarchetti/mimic-iii', '/home/amarchetti')
    data  = MimicData.from_mimic(mimic, pad_visits=False, pad_codes=True)
    print('data loaded')

    model = LinearModel (
        n_embeddings = data.encodings.size + 1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train (
        model     = model,
        optimizer = optimizer,
        data      = data,
        split     = (0, 5000),
        epochs    = 1000,
    )
    print('start testing...')
    result = test (
        model,
        data,
        (6000, len(data.patients)),
        [10, 20, 30],
    )
    print(result)
    print('ok')
    


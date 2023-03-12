from time import sleep

import os
from typing import TypeVar, Tuple
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

RANDOM_STATE = 42 # reproducibility

device = 'cuda'

_Mimic = TypeVar('_Mimic', bound='Mimic') # this should be converted to `Self` in python 3.11
class Mimic:
    """
        holds all the informations about the location of mimic and ccs files
    """
    diagnoses  : str
    admissions : str
    ccs_convert: str

    DEFAULT_DIAGNOSES   = 'DIAGNOSES_ICD.csv.gz'
    DEFAULT_ADMISSIONS  = 'ADMISSIONS.csv.gz'
    DEFAULT_CCS_CONVERT = 'dxicd2ccsxw.csv'

    def __init__(self, *, diagnoses: str, admissions: str, ccs_convert: str):
        self.diagnoses   = diagnoses
        self.admissions  = admissions
        self.ccs_convert = ccs_convert

    def from_folder(folder: str, ccs_convert: str) -> _Mimic:
        """
            returns a Mimic object from given folder with default file names
        """
        diagnoses   = os.path.join(folder,      Mimic.DEFAULT_DIAGNOSES)
        admissions  = os.path.join(folder,      Mimic.DEFAULT_ADMISSIONS)
        ccs_convert = os.path.join(ccs_convert, Mimic.DEFAULT_CCS_CONVERT)
        return Mimic (
            diagnoses   = diagnoses,
            admissions  = admissions,
            ccs_convert = ccs_convert,
        )



_MimicData = TypeVar('_MimicData', bound='MimicData')  # this should be converted to `Self` in python 3.11
class MimicData:
    codes     : np.ndarray
    visits    : np.ndarray
    patients  : np.ndarray
    encodings : np.ndarray

    def __init__(self, *, codes, visits, patients, encodings):
        self.codes     = codes
        self.visits    = visits
        self.patients  = patients
        self.encodings = encodings
    

    def from_mimic(mimic: Mimic, *, pad_visits=False, pad_codes=False) -> _MimicData:
        """
            Load and preprocess data from raw mimic files

            WARNING: codes and encodings are shifted by 1 (code 0 means empy or invalid)
        """
        # load only the columns we are interest to
        diagnoses_cols  = ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']
        admissions_cols = ['HADM_ID', 'ADMITTIME', 'SUBJECT_ID']
        ccs_cols        = ['icd', 'ccs'] 
        diagnoses   = pd.read_csv(mimic.diagnoses,   usecols=diagnoses_cols)
        admissions  = pd.read_csv(mimic.admissions,  usecols=admissions_cols)
        ccs_convert = pd.read_csv(mimic.ccs_convert, usecols=ccs_cols)

        # remove null diagnoses in diagnoses
        num_null = diagnoses.ICD9_CODE.isna().sum()
        if num_null > 0:
            print(f'INFO: removing {num_null} invalid records from diagnoses')
            diagnoses = diagnoses.dropna(subset=['ICD9_CODE'])
        del num_null

        # remove non-referenced admission entries
        ref_mask  = admissions.HADM_ID.isin(diagnoses.HADM_ID)
        num_unref = (~ref_mask).sum()
        if num_unref > 0:
            print(f'INFO: dropping {num_unref} unreferenced records from admissions')
            admissions = admissions[ref_mask]
        del ref_mask, num_unref

        # filter patients with at least two visits
        eligible = admissions.SUBJECT_ID.value_counts()
        eligible = eligible[eligible > 1]
        eligible_mask  = admissions.SUBJECT_ID.isin(eligible.index)
        num_uneligible = (~eligible_mask).sum()
        if num_uneligible > 0:
            print(f'INFO: dropping {num_uneligible} records from admissions of patients with less than two visits')
            admissions     = admissions[eligible_mask]
            diagnoses_mask = diagnoses.SUBJECT_ID.isin(eligible.index)
            num_removing   = (~diagnoses_mask).sum()
            print(f'INFO: dropping {num_removing} records from diagnoses of patients with less than two visits')
            diagnoses = diagnoses[diagnoses_mask]
        # we will use eligble later to shuffle the patients
        del eligible_mask, num_uneligible, num_removing, diagnoses_mask

        # do the conversion for codes
        raw_codes = pd.merge(
            diagnoses.ICD9_CODE,
            ccs_convert[['icd', 'ccs']],
            how      = 'left',
            left_on  = 'ICD9_CODE',
            right_on = 'icd',
        )
        cat_codes = raw_codes.ccs.astype('category')
        codes     = cat_codes.cat.codes + 1  
        encodings = cat_codes.cat.categories
        del raw_codes, cat_codes

        # augment diagnoses infos with time
        admissions['int_time'] = admissions.ADMITTIME.apply(pd.to_datetime).astype(int)
        diagnoses = diagnoses.merge(admissions[['HADM_ID', 'int_time']], on='HADM_ID', how='left')

        # shuffle patients
        unique_patients = pd.Series(eligible.index)
        unique_patients = unique_patients.sample(frac=1.0, random_state=RANDOM_STATE)
        unique_patients = unique_patients.reset_index(drop=True)
        unique_patients = pd.Series(np.arange(unique_patients.size), index=unique_patients, name='tmp')
        diagnoses = diagnoses.join(unique_patients, on='SUBJECT_ID', how='left')
        diagnoses = diagnoses.drop(columns='SUBJECT_ID').rename(columns={'tmp': 'SUBJECT_ID'})
        del unique_patients
        diagnoses.sort_values(by=['SUBJECT_ID', 'int_time'])

        # find lengths of visits and patients histories
        diagnoses = diagnoses[['SUBJECT_ID', 'int_time']]
        visits = diagnoses.groupby(['SUBJECT_ID', 'int_time'], sort=False).size().sort_index().reset_index()
        patients = visits.groupby('SUBJECT_ID', sort=False).int_time.count().sort_index().reset_index()
        visits = visits.drop(columns='SUBJECT_ID')

        # convert to numpy and pad what must be padded
        p_buffer = np.empty((len(patients) + 1), dtype=np.int32)
        patients = patients.int_time.to_numpy()
        p_buffer[1:] = patients
        p_buffer[0] = 0
        np.cumsum(p_buffer, out=p_buffer)

        visits = visits[0].to_numpy(dtype=np.int32)

        codes = codes.to_numpy(dtype=np.int32)
        if pad_codes:
            SIZE = 64
            num_pads = SIZE - visits
            pad = np.zeros(num_pads.sum(), dtype=codes.dtype)
            pad_pos = np.repeat(np.cumsum(visits), num_pads)
            codes = np.insert(codes, pad_pos, pad)
            codes = codes.reshape((-1, SIZE))

        if pad_visits:
            SIZE = 64
            num_pads = SIZE - patients
            pad = np.zeros(num_pads.sum(), dtype=visits.dtype)
            pad_pos = np.repeat(p_buffer[1:], num_pads)
            visits = np.insert(visits, pad_pos, pad)
            visits = visits.reshape((-1, SIZE))
        else:
            buffer = np.empty((len(visits) + 1), dtype=np.int32)
            buffer[1:] = visits
            buffer[0] = 0
            visits = buffer
            np.cumsum(visits, out=visits)

        patients = p_buffer

        return MimicData (
            codes     = codes,
            visits    = visits,
            patients  = patients,
            encodings = encodings.to_numpy(),
        )


class DraiModel(nn.Module):
    hidden_size: int
    embedding  : nn.Module
    sequential : nn.Module
    
    def __init__(self, *, hidden_size: int, n_embeddings: int, n_layers: int, dropout: float):
        # some housekeeping
        super(DraiModel, self).__init__()
        self.hidden_size = hidden_size

        # create the layers
        # softmax is not included since the CrossEntropyLoss already applies it
        self.embedding = nn.Embedding (
            num_embeddings = n_embeddings,
            embedding_dim  = hidden_size,
        )
        self.gru = nn.GRU (
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout,
        )
        self.sequential = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                in_features  = hidden_size,
                out_features = n_embeddings,
            ),
        )

    def forward(self, codes: torch.Tensor, visits: np.ndarray):
        slice_codes = codes[visits[0]:visits[visits.size-1]]
        embeddings = self.embedding(slice_codes)
        buffer = torch.empty((visits.size, self.hidden_size)).to(device)
        last = 0
        for it in range(visits.size-1):
            new = visits[it+1]
            buffer[it] = torch.sum(embeddings[last:new])
            last = new
        output, _hidden = self.gru(buffer)
        output = self.sequential(output)
        return output

    # @todo split should be something more flexible (maybe a (int, int) for the window?)
    def train(self, optimizer: optim.Optimizer, data: MimicData, split: int, epochs: int):
        codes = torch.from_numpy(data.codes).to(device)
        cross_entropy = nn.CrossEntropyLoss(reduction='sum')

        for epoch in range(epochs):
            start = 0
            for it in range(split):
                end = data.patients[it+1]
                visits = data.visits[start: end]
                predictions = self(codes, visits)

                loss = 0
                first_code = data.visits[int(start+1)]
                for i in range(predictions.shape[0]):
                    prediction = predictions[i]
                    last_code = data.visits[int(start+i+2)]
                    for j in range(first_code, last_code):
                        target = codes[j]
                        loss += cross_entropy(prediction, target) / (end - start)
                    first_code = last_code

                loss.backward()
                optimizer.step()

                start = end
            print(f'Iteration {epoch}: done')


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

    def forward(self, codes: torch.Tensor):
        embeddings = self.embedding_bag(codes)
        # predictions = torch.cumsum(embeddings, dim=0)
        return embeddings

def train(
    model     : LinearModel,
    *,
    optimizer : optim.Optimizer,
    data      : MimicData,
    split     : Tuple[int, int],
    epochs    : int,
):
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
        print(f'epoch {epoch} done with loss {train_loss/num_batches}')

def test(
    model: LinearModel,
    data: MimicData,
    split: Tuple[int, int]
) -> float:
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
    cursor = data.patients[split[0]]
    for p in range(split[0]+1, split[1]):
        new_cursor = data.patients[p]
        prediction_cats = codes[cursor:new_cursor-1]
        predictions = model(prediction_cats)

        with torch.no_grad():
            targets_cat = codes[cursor+1: new_cursor]
            targets = target_encoder(targets_cat)
            loss = cross_entropy(predictions, targets)
            train_loss += loss
        cursor = new_cursor

    return total_loss / num_batches
    

if __name__ == '__main__':
    # these paths are relevant for the cnr machine only
    mimic = Mimic.from_folder('/home/amarchetti/mimic-iii', '/home/amarchetti')
    data = MimicData.from_mimic(mimic, pad_visits=False, pad_codes=True)
    print('data loaded')

    model = LinearModel (
        n_embeddings = data.encodings.size + 1,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(
        model     = model,
        optimizer = optimizer,
        data      = data,
        split     = (0, 400),
        epochs    = 1000,
    )
    print('ok')
    


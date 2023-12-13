import polars as pl
import numpy as np

import lib.generator as gen

from kelso_model import *

import torch # @todo necessary?

ontology_path  = 'data/processed/ontology.parquet'
diagnoses_path = 'data/processed/diagnoses.parquet'
ccs_path       = 'data/processed/ccs.parquet'
model_path     = 'results/kelso2-bpjok-2023-11-17_16:05:31'

dataset_split = 100
reference     = 2
k_reals       = 50

def print_patient(ids: np.ndarray, cnt: np.ndarray, ontology: pl.DataFrame):
    codes = pl.DataFrame({'icd9_id':ids})
    codes = codes.join(ontology, how='left', on='icd9_id')

    cursor = 0
    for it in range(cnt.shape[0]):
        length = cnt[it]
        lines = []
        for jt in range(cursor, cursor+length):
            x = f'[{codes["icd_code"][jt]:}]' 
            txt  = f'    {x: <10}'
            txt += f'{codes["label"][jt]}'
            lines.append(txt)
        txt = '\n'.join(lines)
        print(f'visit {it+1}')
        print(txt)
        cursor += length
            

ontology  = pl.read_parquet(ontology_path)
diagnoses = pl.read_parquet(diagnoses_path)
ccs_data  = pl.read_parquet(ccs_path)

diagnoses = diagnoses.head(100_000)

unique_codes = diagnoses['icd9_id'].explode().unique().to_numpy()
max_ccs_id = ccs_data['ccs_id'].max() + 1

ccs_codes = list(diagnoses['ccs_id'  ].to_numpy())
icd_codes = list(diagnoses['icd9_id' ].to_numpy())
positions = list(diagnoses['position'].to_numpy())
counts    = list(diagnoses['count'   ].to_numpy())

ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
gen.create_c2c_table(ontology_array, unique_codes)

distance_list = gen.find_neighbours (
    icd_codes[reference],
    counts[reference],
    icd_codes[dataset_split:],
    counts[dataset_split:],
    0, # this is unused for now
)
topk = np.argpartition(distance_list, k_reals-1)[:k_reals]

model = load_kelso_for_inference(model_path)

batch_codes     = []
batch_counts    = []
batch_positions = []
for it in range(k_reals):
    batch_codes    .append(icd_codes[topk[it]])
    batch_counts   .append(counts   [topk[it]])
    batch_positions.append(positions[topk[it]])

batch = prepare_batch_for_inference(batch_codes, batch_counts, batch_positions, torch.device('cuda'))

result = model(**batch.unpack())

breakpoint()

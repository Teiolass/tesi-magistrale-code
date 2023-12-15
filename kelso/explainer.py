import polars as pl
import numpy as np

import lib.generator as gen

from kelso_model import *

import torch # @todo necessary?
from sklearn.tree import DecisionTreeClassifier

ontology_path  = 'data/processed/ontology.parquet'
diagnoses_path = 'data/processed/diagnoses.parquet'
ccs_path       = 'data/processed/ccs.parquet'
model_path     = 'results/kelso2-bpjok-2023-11-17_16:05:31'

reference     = 2
k_reals       = 500
batch_size    = 64
topk_predictions           = 30
num_top_important_features = 10

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

diagnoses_train = diagnoses.filter(pl.col('role') == 'train')
diagnoses_eval  = diagnoses.filter(pl.col('role') == 'eval')

unique_codes = diagnoses['icd9_id'].explode().unique().to_numpy()
max_ccs_id = ccs_data['ccs_id'].max() + 1

ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
gen.create_c2c_table(ontology_array, unique_codes)

model = load_kelso_for_inference(model_path)

# Extract the numpy data

ccs_codes_train = list(diagnoses_train['ccs_id'  ].to_numpy())
icd_codes_train = list(diagnoses_train['icd9_id' ].to_numpy())
positions_train = list(diagnoses_train['position'].to_numpy())
counts_train    = list(diagnoses_train['count'   ].to_numpy())

ccs_codes_eval = list(diagnoses_eval['ccs_id'  ].to_numpy())
icd_codes_eval = list(diagnoses_eval['icd9_id' ].to_numpy())
positions_eval = list(diagnoses_eval['position'].to_numpy())
counts_eval    = list(diagnoses_eval['count'   ].to_numpy())

# Find closest neighbours

print(f'generating neighbours from a dataset of {len(icd_codes_train)} items')
distance_list = gen.find_neighbours (
    icd_codes_eval[reference],
    counts_eval[reference],
    icd_codes_train,
    counts_train,
    0, # this is unused for now
)
topk = np.argpartition(distance_list, k_reals-1)[:k_reals]

neigh_icd       = []
batch_ccs       = []
neigh_counts    = []
neigh_positions = []
for it in range(k_reals):
    neigh_icd      .append(icd_codes_train[topk[it]])
    batch_ccs      .append(ccs_codes_train[topk[it]])
    neigh_counts   .append(counts_train   [topk[it]])
    neigh_positions.append(positions_train[topk[it]])

# Present result to explain

batch = prepare_batch_for_inference(
    [icd_codes_eval[reference]],
    [counts_eval[reference]],
    [positions_eval[reference]],
    torch.device('cuda'),
)
output = model(**batch.unpack())
output = output[0][-1]
labels = output.topk(topk_predictions).indices.cpu().numpy().astype(np.uint32)

labels_with_description = pl.DataFrame({'ccs_id':labels}).join(ccs_data, how='left', on='ccs_id')
for it, row in enumerate(labels_with_description.iter_rows()):
    selector = f'{it})'
    code = row[1]
    description = row[2]
    code = f'[{code}]' 
    line = f'{selector: >3} {code: <7} {description} - {row[0]}'
    print(line)

index_to_explain = input('What do you want to explain? ')
index_to_explain = int(index_to_explain)
ccs_to_explain = labels[index_to_explain]

# Black Box Predictions

labels = []
cursor = 0
while cursor < len(neigh_icd):
    new_cursor = min(cursor+batch_size, len(neigh_icd))
    batch   = prepare_batch_for_inference (
        neigh_icd      [cursor:new_cursor],
        neigh_counts   [cursor:new_cursor],
        neigh_positions[cursor:new_cursor],
        torch.device('cuda')
    )
    outputs = model(**batch.unpack())
    outputs = [x[-1] for x in outputs]
    # @todo I should pack these lists
    batch_labels = [x.topk(topk_predictions).indices.cpu().numpy() for x in outputs]
    batch_labels = [np.any(x == ccs_to_explain) for x in batch_labels]

    labels += batch_labels
    cursor = new_cursor

# Tree fitting

tree_inputs = gen.ids_to_encoded(batch_ccs, neigh_counts, max_ccs_id, 0.5)

# @todo add appropriate args
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(tree_inputs, labels)

# Extract explanation

feature_importances = tree_classifier.feature_importances_
top_important_features = np.argpartition(- np.abs(feature_importances), range(num_top_important_features))
top_important_features = top_important_features[:num_top_important_features]
importances = feature_importances[top_important_features]

# Present the result to the user

df = pl.DataFrame({'ccs_id':top_important_features.astype(np.uint32), 'importance':importances})
df = df.join(ccs_data, how='left', on='ccs_id')

for it, row in enumerate(df.iter_rows()):
    importance = row[1]
    code = row[2]
    description = row[3]

    pos  = f'{it+1})'
    code = f'[{code}]'
    line = f'{pos: <4}{importance: <7.3f} {code: <8} {description}'
    print(line)


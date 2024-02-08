import polars as pl
import numpy as np

import lib.generator as gen

from kelso_model import *
import tomlkit

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import random

from tqdm import tqdm

ontology_path   = 'data/processed/ontology.parquet'
diagnoses_path  = 'data/processed/diagnoses.parquet'
generation_path = 'data/processed/generation.parquet'
ccs_path        = 'data/processed/ccs.parquet'
icd_path        = 'data/processed/icd.parquet'

config_path = 'repo/kelso/explain_config.toml'
output_path = 'results/explainer.txt'

model_path      = 'results/kelso2-dejlv-2024-01-20_17:23:33/'
filler_path     = 'results/filler-xyxdp-2024-01-21_15:16:51/'
k_reals          = 50
batch_size       = 64
keep_prob        = 0.8
topk_predictions = 10
ontological_perturbation   = True
generative_perturbation    = True
uniform_perturbation       = False
tree_train_fraction        = 0.75
synthetic_multiply_factor  = 4
generative_multiply_factor = 4
filter_codes_present       = True

reference_index = 0

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
            
def explain_label(neigh_ccs, neigh_counts, labels, max_ccs_id, tree_train_fraction, target):
    # Tree fitting
    tree_inputs = gen.ids_to_encoded(neigh_ccs, neigh_counts, max_ccs_id, 0.5)

    # @todo add appropriate args
    tree_classifier = DecisionTreeClassifier()

    train_split = int(tree_train_fraction * len(labels))

    tree_inputs_train = tree_inputs[:train_split]
    tree_inputs_eval  = tree_inputs[train_split:]
    labels_train      = labels[:train_split, target]
    labels_eval       = labels[train_split:, target]

    tree_classifier.fit(tree_inputs_train, labels_train)

    return tree_classifier

# Load base data and model

ontology  = pl.read_parquet(ontology_path)
diagnoses = pl.read_parquet(diagnoses_path)
ccs_data  = pl.read_parquet(ccs_path)

diagnoses_train = diagnoses.filter(pl.col('role') == 'train')
diagnoses_eval  = diagnoses.filter(pl.col('role') == 'eval')

unique_codes = diagnoses['icd9_id'].explode().unique().to_numpy()
max_ccs_id = ccs_data['ccs_id'].max() + 1

ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
gen.create_c2c_table(ontology_array, unique_codes)


# Extract the numpy data

ccs_codes_train = list(diagnoses_train['ccs_id'  ].to_numpy())
icd_codes_train = list(diagnoses_train['icd9_id' ].to_numpy())
positions_train = list(diagnoses_train['position'].to_numpy())
counts_train    = list(diagnoses_train['count'   ].to_numpy())

ccs_codes_eval = list(diagnoses_eval['ccs_id'  ].to_numpy())
icd_codes_eval = list(diagnoses_eval['icd9_id' ].to_numpy())
positions_eval = list(diagnoses_eval['position'].to_numpy())
counts_eval    = list(diagnoses_eval['count'   ].to_numpy())


model = load_kelso_for_inference(model_path)

total_accuracy = 0.0
total_f1_score = 0.0

num_layers = len(model.model.decoder_layers) 
importance_analysis = ['importance', 'max', 'min', 'avg']
importance_source = ['all'] + ['layer_{}'.format(x) for x in range(num_layers)]
correlation_matrices = {s: np.zeros((4,4)) for s in importance_source}


if generative_perturbation:
    filler, hole_prob, hole_token_id = load_kelso_for_generation(filler_path)
    conv_data = pl.read_parquet(generation_path).sort('out_id')
    zero_row  = pl.DataFrame({'icd9_id':0, 'out_id':0, 'ccs_id':0}, schema=conv_data.schema)
    conv_data = pl.concat([zero_row, conv_data])

    out_to_icd = conv_data['icd9_id'].to_numpy()
    out_to_ccs = conv_data['ccs_id' ].to_numpy()

# Find closest neighbours in the real data
distance_list = gen.compute_patients_distances (
    icd_codes_eval[reference_index],
    counts_eval[reference_index],
    icd_codes_train,
    counts_train,
    0, # this is unused for now
)
topk = np.argpartition(distance_list, k_reals-1)[:k_reals]

neigh_icd       = []
neigh_ccs       = []
neigh_counts    = []
neigh_positions = []
for it in range(k_reals):
    neigh_icd      .append(icd_codes_train[topk[it]])
    neigh_ccs      .append(ccs_codes_train[topk[it]])
    neigh_counts   .append(counts_train   [topk[it]])
    neigh_positions.append(positions_train[topk[it]])

neigh_icd      .append(icd_codes_eval[reference_index])
neigh_ccs      .append(ccs_codes_eval[reference_index])
neigh_counts   .append(counts_eval   [reference_index])
neigh_positions.append(positions_eval[reference_index])


# augment the neighbours with some synthetic points

if ontological_perturbation:
    displacements, new_counts = gen.ontological_perturbation(neigh_icd, neigh_counts, synthetic_multiply_factor, keep_prob)

    new_neigh_icd       = []
    new_neigh_ccs       = []
    new_neigh_positions = []
    for it, (icd, ccs, pos) in enumerate(zip(neigh_icd, neigh_ccs, neigh_positions)):
        for jt in range(synthetic_multiply_factor):
            displ = displacements[synthetic_multiply_factor * it + jt]
            new_neigh_icd      .append(icd[displ])
            new_neigh_ccs      .append(ccs[displ])
            new_neigh_positions.append(pos[displ])
    neigh_icd       += new_neigh_icd
    neigh_ccs       += new_neigh_ccs
    neigh_positions += new_neigh_positions
    neigh_counts += new_counts

if generative_perturbation:
    new_neigh_icd       = []
    new_neigh_ccs       = []
    new_neigh_counts    = []
    new_neigh_positions = []
    cursor = 0
    while cursor < len(neigh_icd):
        new_cursor = min(cursor + batch_size, len(neigh_icd))

        for _ in range(generative_multiply_factor):
            batch = prepare_batch_for_generation (
                neigh_icd      [cursor:new_cursor],
                neigh_counts   [cursor:new_cursor],
                neigh_positions[cursor:new_cursor],
                hole_prob,
                hole_token_id,
                torch.device('cuda')
            )

            if uniform_perturbation:
                bsz = batch.codes.shape[0]
                b_n = batch.codes.shape[1]
                n_out = filler.head.out_features
                gen_output = torch.zeros((bsz, b_n, n_out))
            else:
                gen_output = filler(**batch.unpack()) # (bsz, b_n, n_out)
            old_shape = gen_output.shape
            gen_output = gen_output.reshape((-1, gen_output.shape[-1]))
            gen_output = torch.softmax(gen_output, dim=-1)

            new_codes = torch.multinomial(gen_output, 1)
            new_codes = new_codes.reshape(old_shape[:-1])
            new_codes = new_codes.cpu().numpy()

            new_icd = list(out_to_icd[new_codes])
            new_ccs = list(out_to_ccs[new_codes])

            new_neigh_icd       += new_icd
            new_neigh_ccs       += new_ccs
            new_neigh_counts    += neigh_counts[cursor:new_cursor]
            new_neigh_positions += neigh_positions[cursor:new_cursor]

        cursor = new_cursor
    neigh_icd       += new_neigh_icd
    neigh_ccs       += new_neigh_ccs
    neigh_counts    += new_neigh_counts
    neigh_positions += new_neigh_positions


batch = prepare_batch_for_inference(
    [icd_codes_eval[reference_index]],
    [counts_eval[reference_index]],
    [positions_eval[reference_index]],
    torch.device('cuda'),
)
# attentions has shape (1, num_layers, num_heads, b_n, b_n)
output, attention = model(**batch.unpack(), return_attention=True)
output = output[0][-1]
labels = output.topk(topk_predictions).indices

arr_present_ccs = ccs_codes_eval[reference_index]
present_ccs = np.zeros((max_ccs_id,), dtype=bool)
for i in range(len(arr_present_ccs)):
    present_ccs[arr_present_ccs[i]] = True

attention = attention.squeeze(0)
attention_all = analyze_attention(attention, reference_index)
attention_layers = [analyze_attention(attention[x], reference_index) for x in range(attention.shape[0])]

# Black Box Predictions

neigh_labels = np.empty((len(labels), len(neigh_icd), ), dtype=np.bool_)
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
    outputs = torch.stack(outputs)
    batch_labels = outputs.topk(topk_predictions, dim=-1).indices
    batch_labels = (batch_labels == labels[:,None,None]).any(-1)
    batch_labels = batch_labels.cpu().numpy()

    neigh_labels[:, cursor:new_cursor] = batch_labels
    cursor = new_cursor
neigh_labels = neigh_labels.transpose()

tree = explain_label (
    neigh_ccs, neigh_counts, neigh_labels, max_ccs_id, tree_train_fraction, list(range(topk_predictions))
)

reference_enc = gen.ids_to_encoded(
    [ccs_codes_eval[reference_index]],
    [counts_eval[reference_index]],
    max_ccs_id,
    0.5
)[0]


tree_path = tree.tree_.decision_path(reference_enc.reshape((1,-1))).indices
features = tree.tree_.feature

expl_labels = [features[i] for i in tree_path]
expl_labels = [x for x in expl_labels if x >= 0]

thresholds = tree.tree_.threshold
thresholds = [thresholds[i] for i in tree_path if features[i] >= 0]

df = pl.DataFrame({'ccs_id': expl_labels, 'threasholds': thresholds}).with_columns(ccs_id=pl.col('ccs_id').cast(pl.UInt32))
df = df.join(ccs_data, on='ccs_id', how='left')

print_patient(icd_codes_eval[reference_index], counts_eval[reference_index], ontology)
print('\n')

print('ccs predicted')
labels = labels.tolist()
for id in labels:
    infos = ccs_data.filter(pl.col('ccs_id') == id)
    if len(infos) != 1:
        raise ValueError('should not happen')
    
    code = infos['ccs'][0]
    label = infos['description'][0]

    print(f'{"["+str(code)+"]": <7} {label}')


print('\ndecision rules')
for (id, thresh, ccs, desc) in df.iter_rows():
    txt = f'code {ccs: <4} - threashold: {thresh:.2f}, found {reference_enc[id]:.2f},  ({desc})'
    print(txt)

print('\ncodes of the patient relevant for explanation')
for id in expl_labels:
    for it in range(len(ccs_codes_eval[reference_index])):
        ccs = ccs_codes_eval[reference_index][it]
        if ccs == id:
            visit = positions_eval[reference_index][it] + 1
            icd   = icd_codes_eval[reference_index][it] + 1

            infos = ontology.filter(pl.col('icd9_id') == icd)
            if len(infos) != 1:
                raise ValueError('Should not happen')
            code = infos['icd_code'][0]
            label = infos['label'][0]

            print(f'At visit {(str(visit)+","): <3} code {code: <6} [{label}]')


import polars as pl
import numpy as np

import lib.generator as gen

from kelso_model import *

import torch # @todo necessary?
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import random

from tqdm import tqdm

ontology_path  = 'data/processed/ontology.parquet'
diagnoses_path = 'data/processed/diagnoses.parquet'
ccs_path       = 'data/processed/ccs.parquet'
model_path     = 'results/a-kelso2-xjdmk-2023-12-16_15:37:22'

k_reals        = 500
batch_size     = 64
keep_prob      = 0.8
num_references = 10
topk_predictions           = 30
tree_train_fraction        = 0.8
num_top_important_features = 10
synthetic_multiply_factor  = 10

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
            
def choose_ccs_to_explain(labels: np.ndarray, ccs_data: pl.DataFrame) -> int:
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
    return ccs_to_explain

def explain_label(neigh_ccs, neigh_counts, labels, max_ccs_id, tree_train_fraction):
    # Tree fitting

    tree_inputs = gen.ids_to_encoded(neigh_ccs, neigh_counts, max_ccs_id, 0.5)

    # @todo add appropriate args
    tree_classifier = DecisionTreeClassifier()

    train_split = int(tree_train_fraction * len(labels))

    tree_inputs_train = tree_inputs[:train_split]
    tree_inputs_eval  = tree_inputs[train_split:]
    labels_train      = labels[:train_split]
    labels_eval       = labels[train_split:]

    tree_classifier.fit(tree_inputs_train, labels_train)
    outputs = tree_classifier.predict(tree_inputs_eval)
    # @todo
    accuracy = metrics.accuracy_score(labels_eval, outputs)
    f1_score = metrics.f1_score(labels_eval, outputs)

    # Extract explanation

    feature_importances = tree_classifier.feature_importances_
    top_important_features = np.argpartition(- np.abs(feature_importances), range(num_top_important_features))
    top_important_features = top_important_features[:num_top_important_features]
    importances = feature_importances[top_important_features]

    return top_important_features, importances, accuracy, f1_score


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

total_accuracy = 0.0
total_f1_score = 0.0

for reference in tqdm(range(num_references), leave=False):
    # Find closest neighbours in the real data
    distance_list = gen.compute_patients_distances (
        icd_codes_eval[reference],
        counts_eval[reference],
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

    # augment the neighbours with some synthetic points

    displacements, new_counts = gen.independent_perturbation(neigh_icd, neigh_counts, synthetic_multiply_factor, keep_prob)

    neigh_counts = new_counts
    new_neigh_icd       = []
    new_neigh_ccs       = []
    new_neigh_counts    = []
    new_neigh_positions = []
    for it, (icd, ccs, pos) in enumerate(zip(neigh_icd, neigh_ccs, neigh_positions)):
        for jt in range(synthetic_multiply_factor):
            displ = displacements[synthetic_multiply_factor * it + jt]
            new_neigh_icd      .append(icd[displ])
            new_neigh_ccs      .append(ccs[displ])
            new_neigh_positions.append(pos[displ])
    neigh_icd       = new_neigh_icd
    neigh_ccs       = new_neigh_ccs
    neigh_positions = new_neigh_positions

    # Choose result to explain

    batch = prepare_batch_for_inference(
        [icd_codes_eval[reference]],
        [counts_eval[reference]],
        [positions_eval[reference]],
        torch.device('cuda'),
    )
    output = model(**batch.unpack())
    output = output[0][-1]
    labels = output.topk(topk_predictions).indices.cpu().numpy().astype(np.uint32)

    # @debug
    # ccs_to_explain = choose_ccs_to_explain(labels, ccs_data)
    # @debug
    id_to_explain = random.randrange(0, len(labels))
    ccs_to_explain = labels[id_to_explain]

    # Black Box Predictions

    labels = np.empty((len(neigh_icd), ), dtype=np.bool_)
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
        if any([len(x) == 0 for x in outputs]):
            breakpoint()
        outputs = [x[-1] for x in outputs]
        outputs = torch.stack(outputs)
        batch_labels = outputs.topk(topk_predictions, dim=-1).indices
        batch_labels = torch.any(batch_labels == ccs_to_explain, dim=-1)
        batch_labels = batch_labels.cpu().numpy()

        labels[cursor:new_cursor] = batch_labels
        cursor = new_cursor

    top_important_features, importances, accuracy, f1_score = explain_label(neigh_ccs, neigh_counts, labels, max_ccs_id, tree_train_fraction)
    # @debug
    # print(f'accuracy is {accuracy*100:.2f}%')

    # @debug
    total_accuracy += accuracy
    total_f1_score += f1_score

    # Present the result to the user

    df = pl.DataFrame({'ccs_id':top_important_features.astype(np.uint32), 'importance':importances})
    df = df.join(ccs_data, how='left', on='ccs_id')

    # @debug
    # for it, row in enumerate(df.iter_rows()):
    #     importance = row[1]
    #     code = row[2]
    #     description = row[3]

    #     pos  = f'{it+1})'
    #     code = f'[{code}]'
    #     line = f'{pos: <4}{importance: <7.3f} {code: <8} {description}'
    #     print(line)

accuracy = total_accuracy / num_references
f1_score = total_f1_score / num_references
print(f'avg accuracy: {accuracy*100:.4f}%')
print(f'avg f1_score: {f1_score*100:.4f}%')

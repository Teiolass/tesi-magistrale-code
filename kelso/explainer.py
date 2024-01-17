import polars as pl
import numpy as np

import lib.generator as gen

from kelso_model import *

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import random

from tqdm import tqdm

ontology_path   = 'data/processed/ontology.parquet'
diagnoses_path  = 'data/processed/diagnoses.parquet'
generation_path = 'data/processed/generation.parquet'
ccs_path        = 'data/processed/ccs.parquet'
model_path      = 'results/kelso2-gopaf-2023-12-22_10:11:59/'
filler_path     = ''

k_reals          = 200
batch_size       = 64
keep_prob        = 0.8
num_references   = 20
topk_predictions = 30
augment_neighbours         = False
generative_perturbation    = True
tree_train_fraction        = 0.5
num_top_important_features = 10
synthetic_multiply_factor  = 4
generative_multiply_factor = 4

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

def explain_all_labels(neigh_ccs, neigh_counts, labels, max_ccs_id, tree_train_fraction):
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
    accuracy = metrics.accuracy_score(labels_eval.flatten(), outputs.flatten())
    f1_score = metrics.f1_score(labels_eval, outputs, average='micro')

    # Extract explanation

    feature_importances = tree_classifier.feature_importances_

    # return top_important_features, importances, accuracy, f1_score
    return feature_importances, accuracy, f1_score


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

importance_ds = {
    'tree_importance': [],
    'attention_max': [],
    'attention_min': [],
    'attention_avg': [],
}


if generative_perturbation:
    filler, hole_prob, hole_token_id = load_kelso_for_generation(filler_path)
    conv_data = pl.read_parquet(generation_path).sort('out_id')
    zero_row  = pl.DataFrame({'icd9_id':0, 'out_id':0, 'ccs_id':0}, schema=conv_data.schema)
    conv_data = pl.concat([zero_row, conv_data])

    out_to_icd = conv_data['icd9_id'].to_numpy()
    out_to_ccs = conv_data['ccs_id' ].to_numpy()

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

    if augment_neighbours:
        displacements, new_counts = gen.ontological_perturbation(neigh_icd, neigh_counts, synthetic_multiply_factor, keep_prob)

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

    if generative_perturbation:
        new_neigh_icd       = []
        new_neigh_ccs       = []
        new_neigh_counts    = []
        new_neigh_positions = []
        cursor = 0
        while cursor < len(neigh_icd):
            new_cursor = min(cursor + batch_size, len(neigh_icd))

            batch = prepare_batch_for_generation (
                neigh_icd      [cursor:new_cursor],
                neigh_counts   [cursor:new_cursor],
                neigh_positions[cursor:new_cursor],
                hole_prob,
                hole_token_id,
                torch.device('cuda')
            )

            gen_output = filler(**batch.unpack()) # (bsz, b_n, n_out)
            old_shape = gen_output.shape
            gen_output = gen_output.reshape((-1, gen_output.shape[-1]))
            gen_output = torch.softmax(gen_output, dim=-1)

            for _ in range(generative_multiply_factor):
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

    # Choose result to explain

    batch = prepare_batch_for_inference(
        [icd_codes_eval[reference]],
        [counts_eval[reference]],
        [positions_eval[reference]],
        torch.device('cuda'),
    )
    output, attention = model(**batch.unpack(), return_attention=True)
    output = output[0][-1]
    labels = output.topk(topk_predictions).indices

    attention = attention.squeeze(0)
    attention_flat = attention.reshape((-1, attention.shape[-1]))
    attention_max_seq = attention_flat.max(0).values
    attention_avg_seq = attention_flat.mean(0)
    attention_flat_filtered = attention_flat
    attention_flat_filtered[attention_flat_filtered < 1e-6] = 2.00
    attention_min_seq = attention_flat_filtered.min(0).values
    attention_max = np.zeros((max_ccs_id,))
    attention_min = np.ones((max_ccs_id,))
    attention_avg = np.zeros((max_ccs_id,))
    attention_cnt = np.zeros((max_ccs_id,))
    if attention_flat.shape[-1] != len(ccs_codes_eval[reference]):
        raise ValueError('Mismatch in sizes')
    for it in range(len(ccs_codes_eval[reference])):
        c = ccs_codes_eval[reference][it]
        attention_max[c] = max(attention_max[c], float(attention_max_seq[it]))
        attention_min[c] = min(attention_min[c], float(attention_min_seq[it]))
        attention_avg[c] = attention_avg[c] + float(attention_avg_seq[it])
        attention_cnt[c] += 1
    attention_avg = attention_avg / np.maximum(attention_cnt, 1)

    # @debug
    # ccs_to_explain = choose_ccs_to_explain(labels, ccs_data)
    # @debug
    # id_to_explain = random.randrange(0, len(labels))

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

    # top_important_features, importances, accuracy, f1_score = explain_label(neigh_ccs, neigh_counts, neigh_labels[:, id_to_explain], max_ccs_id, tree_train_fraction)
    importances, accuracy, f1_score = explain_all_labels(neigh_ccs, neigh_counts, neigh_labels, max_ccs_id, tree_train_fraction)
    # @debug
    # print(f'accuracy is {accuracy*100:.2f}%')

    importance_ds['tree_importance'].append(importances)
    importance_ds['attention_max'].append(attention_max)
    importance_ds['attention_min'].append(attention_min)
    importance_ds['attention_avg'].append(attention_avg)

    # @debug
    total_accuracy += accuracy
    total_f1_score += f1_score

    # Present the result to the user

    # df = pl.DataFrame({'ccs_id':top_important_features.astype(np.uint32), 'importance':importances})
    # df = df.join(ccs_data, how='left', on='ccs_id')

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

importance_ds['tree_importance'] = np.concatenate(importance_ds['tree_importance'])
importance_ds['attention_max']   = np.concatenate(importance_ds['attention_max'])
importance_ds['attention_min']   = np.concatenate(importance_ds['attention_min'])
importance_ds['attention_avg']   = np.concatenate(importance_ds['attention_avg'])

breakpoint()



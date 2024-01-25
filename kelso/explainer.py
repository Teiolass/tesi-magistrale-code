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

config_path = 'repo/kelso/explain_config.toml'
output_path = 'results/explainer.txt'

# model_path      = 'results/kelso2-dejlv-2024-01-20_17:23:33/'
# filler_path     = 'results/filler-xyxdp-2024-01-21_15:16:51/'
# k_reals          = 200
# batch_size       = 64
# keep_prob        = 0.8
# num_references   = 500
# topk_predictions = 30
# ontological_perturbation   = False
# generative_perturbation    = True
# uniform_perturbation       = True
# tree_train_fraction        = 0.75
# num_top_important_features = 10
# synthetic_multiply_factor  = 4
# generative_multiply_factor = 16
# filter_codes_present       = True

def analyze_attention(attention: torch.Tensor, reference: int) -> dict[str, torch.Tensor]:
    attention_flat = attention.reshape((-1, attention.shape[-1]))
    attention_max_seq = attention_flat.max(0).values
    attention_avg_seq = attention_flat.mean(0)
    attention_flat_filtered = attention_flat.clone()
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

    return {
        'max': attention_max,
        'min': attention_min,
        'avg': attention_avg,
    }


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


# Extract the numpy data

ccs_codes_train = list(diagnoses_train['ccs_id'  ].to_numpy())
icd_codes_train = list(diagnoses_train['icd9_id' ].to_numpy())
positions_train = list(diagnoses_train['position'].to_numpy())
counts_train    = list(diagnoses_train['count'   ].to_numpy())

ccs_codes_eval = list(diagnoses_eval['ccs_id'  ].to_numpy())
icd_codes_eval = list(diagnoses_eval['icd9_id' ].to_numpy())
positions_eval = list(diagnoses_eval['position'].to_numpy())
counts_eval    = list(diagnoses_eval['count'   ].to_numpy())

def explain(config):

    model_path                 = config['model_path']
    filler_path                = config['filler_path']
    k_reals                    = config['k_reals']
    batch_size                 = config['batch_size']
    keep_prob                  = config['keep_prob']
    num_references             = config['num_references']
    topk_predictions           = config['topk_predictions']
    ontological_perturbation   = config['ontological_perturbation']
    generative_perturbation    = config['generative_perturbation']
    uniform_perturbation       = config['uniform_perturbation']
    tree_train_fraction        = config['tree_train_fraction']
    num_top_important_features = config['num_top_important_features']
    synthetic_multiply_factor  = config['synthetic_multiply_factor']
    generative_multiply_factor = config['generative_multiply_factor']
    filter_codes_present       = config['filter_codes_present']

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

        if ontological_perturbation:
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

        # Choose result to explain

        batch = prepare_batch_for_inference(
            [icd_codes_eval[reference]],
            [counts_eval[reference]],
            [positions_eval[reference]],
            torch.device('cuda'),
        )
        # attentions has shape (1, num_layers, num_heads, b_n, b_n)
        output, attention = model(**batch.unpack(), return_attention=True)
        output = output[0][-1]
        labels = output.topk(topk_predictions).indices

        arr_present_ccs = ccs_codes_eval[reference]
        present_ccs = np.zeros((max_ccs_id,), dtype=bool)
        for i in range(len(arr_present_ccs)):
            present_ccs[arr_present_ccs[i]] = True

        attention = attention.squeeze(0)
        attention_all = analyze_attention(attention, reference)
        attention_layers = [analyze_attention(attention[x], reference) for x in range(attention.shape[0])]

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

        importances, accuracy, f1_score = explain_all_labels(
            neigh_ccs, neigh_counts, neigh_labels, max_ccs_id, tree_train_fraction
        )

        total_accuracy += accuracy
        total_f1_score += f1_score

        with np.errstate(invalid='raise'):
            m = [importances] + [attention_all[x] for x in importance_analysis[1:]]
            m = np.stack(m)
            if filter_codes_present:
                m = m[:, present_ccs]
            if m.shape[1] == 1:
                correlation_matrices['all'] += np.ones((4, 4))
            else:
                try:
                    correlation_matrices['all'] += np.corrcoef(m)
                except:
                    breakpoint()
            for l in range(num_layers):
                m = [importances] + [attention_layers[l][x] for x in importance_analysis[1:]]
                m = np.stack(m)
                if filter_codes_present:
                    m = m[:, present_ccs]
                if m.shape[1] == 0:
                    breakpoint()
                if m.shape[1] == 1:
                    correlation_matrices[f'layer_{l}'] += np.ones((4, 4))
                else:
                    try:
                        correlation_matrices[f'layer_{l}'] += np.corrcoef(m)
                    except:
                        breakpoint()


    accuracy = total_accuracy / num_references
    f1_score = total_f1_score / num_references
    for source in correlation_matrices:
        correlation_matrices[source] /= num_references

    # print(f'avg accuracy: {accuracy*100:.4f}%')
    # print(f'avg f1_score: {f1_score*100:.4f}%')
    # for source in correlation_matrices:
    #     print(source)
    #     print(correlation_matrices[source])
    #     print()

    return f1_score, correlation_matrices


if __name__ == '__main__':
    with open(config_path, 'r') as f:
        txt = f.read()
    all_config = tomlkit.parse(txt)

    for config in tqdm(all_config.values(), 'config', leave=False):
        f1_score, cms = explain(config)

        vals = []
        vals += [str(config['k_reals'])]
        vals += ['Yes' if config['ontological_perturbation'] else 'No ']
        vals += ['Yes' if config['generative_perturbation']  else 'No ']
        vals += ['Yes' if config['uniform_perturbation']  else 'No ']
        vals += [str(round(f1_score*100, 1)) + '%']

        lines = []
        for k in cms.keys():
            mat = cms[k]
            cc = [mat[0, x] for x in range(1, 4)]
            nvals = vals + [f'{k: <10}'] + [str(round(x, 2)) for x in cc]
            txt = ' & '.join(nvals)
            txt += '  \\\\'
            print(txt)
            lines.append(txt)
        lines = [x + '\n' for x in lines]
        lines = ''.join(lines)
        with open(output_path, 'a') as f:
            f.write(lines)



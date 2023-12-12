import polars as pl
import numpy as np

import lib.generator as gen

ontology_path  = 'data/processed/ontology.parquet'
diagnoses_path = 'data/processed/diagnoses.parquet'
ccs_path       = 'data/processed/ccs.parquet'

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


ccs_codes = list(diagnoses['ccs_id'].to_numpy())
icd_codes = list(diagnoses['icd9_id'].to_numpy())
counts    = list(diagnoses['count'  ].to_numpy())

ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
gen.create_c2c_table(ontology_array, unique_codes)

first = 100
reference = 2
k = 50

result = gen.find_neighbours(icd_codes[reference], counts[reference], icd_codes[first:], counts[first:], 2);
topk = np.argpartition(result, k-1)[:k]



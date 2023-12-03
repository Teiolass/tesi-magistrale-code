import polars as pl
import numpy as np

import lib.generator as gen

ontology_path  = 'data/processed/ontology.parquet'
diagnoses_path = 'data/processed/diagnoses.parquet'

ontology  = pl.read_parquet(ontology_path)
diagnoses = pl.read_parquet(diagnoses_path)

diagnoses = diagnoses.head(4)

codes  = list(diagnoses['icd9_id'].to_numpy())
counts = list(diagnoses['count'  ].to_numpy())

ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
gen.set_ontology(ontology_array)

result = gen.find_neighbours(codes[0], counts[0], codes[1:], counts[1:], 2);

# example_p_cod = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32)
# example_p_cnt = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint32)
# example_c_cod = np.array([2, 4, 5], dtype=np.uint32)
# example_c_cnt = np.array([1, 1, 1], dtype=np.uint32)
# result = gen.find_neighbours(example_p_cod, example_p_cnt, [example_c_cod], [example_c_cnt], 0)

print(result)

import polars as pl

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

gen.find_neighbours(codes[0], counts[0], codes[1:], counts[1:], 2);

breakpoint()

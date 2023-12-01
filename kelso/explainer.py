import polars as pl

import lib.generator as gen

ontology_path = 'data/processed/ontology.parquet'
ontology = pl.read_parquet(ontology_path)

ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
gen.get_parent(ontology_array, 34)

breakpoint()

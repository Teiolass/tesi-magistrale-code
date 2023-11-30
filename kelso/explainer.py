import polars as pl

ontology_path = 'data/processed/ontology.parquet'
ontology = pl.read_parquet(ontology_path)
breakpoint()

ontology_array = ontology.filter(pl.col('parent_id').is_null())['parent_id'].to_numpy()
breakpoint()

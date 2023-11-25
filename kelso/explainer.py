import polars as pl

ontology_path = 'data/ICD9CM.csv'
icd_path = 'data/processed/icd.parquet'

prefixes = ['http://purl.bioontology.org/ontology/ICD9CM/', 'http://purl.bioontology.org/ontology/STY/']

ontology = pl.read_csv(ontology_path)

def remove_prefixes(exp: pl.Expr, prefixes: list[str]) -> pl.Expr:
    for p in prefixes:
        exp = exp.str.strip_prefix(p)
    return exp

ontology = ontology.lazy().select(
    old = remove_prefixes(pl.col('Class ID'), prefixes),
    icd_code = remove_prefixes(pl.col('Class ID'), prefixes).str.replace('.', '', literal=True),
    parent   = remove_prefixes(pl.col('Parents' ), prefixes).str.replace('.', '', literal=True),
).collect()

codes = pl.read_parquet(icd_path)

y = pl.concat([codes['icd_code'], ontology['icd_code']])
breakpoint()

last_leaf_id = codes['icd9_id'].max()


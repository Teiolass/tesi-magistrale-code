import polars as pl
from polars import col

import os

mimic_prefix  = 'data/mimic-iv/2.2/hosp/'
ccs_prefix    = 'data/ccs'
output_prefix = 'data/processed'

diagnoses_file  = 'diagnoses_icd.csv.gz'
admissions_file = 'admissions.csv.gz'
ccs_single_file = 'ccs_single_dx_tool_2015.csv'
icd_conv_file   = 'icd10cmtoicd9gem.csv'
ontology_file   = 'data/ICD9CM.csv'

output_diagnoses = 'diagnoses.parquet'
output_ccs       = 'ccs.parquet'
output_icd       = 'icd.parquet'
output_ontology  = 'ontology.parquet'

min_icd_occurences = 20

ontology_prefixes = ['http://purl.bioontology.org/ontology/ICD9CM/', 'http://purl.bioontology.org/ontology/STY/']
ontology_root_name = 'root'

diagnoses_path = os.path.join(mimic_prefix, diagnoses_file)
diagnoses      = pl.read_csv(diagnoses_path, dtypes={'subject_id':pl.UInt64, 'icd_version':pl.UInt8})
admission_path = os.path.join(mimic_prefix, admissions_file)
admissions     = pl.read_csv(admission_path)
icd_conv_path  = os.path.join(ccs_prefix, icd_conv_file)
icd_conv       = pl.read_csv(icd_conv_path, dtypes={'icd10cm':pl.Utf8, 'icd9cm':pl.Utf8})
ccs_conv_path  = os.path.join(ccs_prefix, ccs_single_file)
ccs_conv       = pl.read_csv(ccs_conv_path, quote_char="'")

diagnoses = diagnoses[['subject_id', 'hadm_id', 'icd_code', 'icd_version']]
diagnoses = diagnoses.with_columns (
    visit_count = col('hadm_id').unique().count().over('subject_id')
).filter(col('visit_count') > 1)


icd_conv = icd_conv.lazy().select (
    col('icd10cm'),
    col('icd9cm').first().over('icd10cm') # @todo maybe we can have a better strategy
).unique().collect()

# convert icd10 to icd9
diagnoses_icd10 = (
    diagnoses
        .filter(col('icd_version') == 10)
        .join(icd_conv, left_on='icd_code', right_on='icd10cm', how='left')
        .select(col('subject_id'), col('hadm_id'), col('icd9cm').alias('icd_code'))
)
diagnoses_icd9 = diagnoses.filter(col('icd_version') == 9)[diagnoses_icd10.columns]
diagnoses = pl.concat([diagnoses_icd9, diagnoses_icd10], how='vertical')

# convert icd9 to ccs
ccs_conv = ccs_conv.select (
    icd9 = col('ICD-9-CM CODE').str.strip_chars(),
    ccs  = col('CCS CATEGORY' ).str.strip_chars().cast(pl.UInt16),
    description = col('CCS CATEGORY DESCRIPTION')
)
diagnoses = (
    diagnoses
    .join(
        ccs_conv[['icd9', 'ccs']],
        left_on  = 'icd_code',
        right_on = 'icd9',
        how = 'left'
    )
    .with_columns (
        ccs = col('ccs').fill_null(pl.lit(0)), # @todo this is a lazy way to deal with null values
        icd_code = col('icd_code').fill_null(pl.lit('NoDx'))
    )
).unique()

# convert ccs codes to ids
ccs_codes = diagnoses[['ccs']].unique().sort('ccs')
we_have_zero_code = 0 in ccs_codes['ccs'].head(1)
if not we_have_zero_code:
    print('WARNING: we do not have a zero code!')
ccs_codes = ccs_codes.join(ccs_conv.drop('icd9').unique('ccs'), on='ccs', how='left')
starting_index = 0 if we_have_zero_code else 1
indexes = pl.DataFrame({'ccs_id': range(starting_index, starting_index+ccs_codes.shape[0])}, schema={'ccs_id':pl.UInt32})
ccs_codes = pl.concat([ccs_codes, indexes], how='horizontal')
diagnoses = diagnoses.join(ccs_codes[['ccs', 'ccs_id']], on='ccs', how='left').drop('ccs')

# convert icd9 codes to id
icd9_codes = diagnoses['icd_code'].value_counts()
num_codes_before = icd9_codes.shape[0]
icd9_codes = icd9_codes.filter(col('counts') >= min_icd_occurences).drop('counts')
print(
    f'There were {num_codes_before} different icd codes in the dataset. '
    f'We are keeping only those with at least {min_icd_occurences}. '
    f'There are {icd9_codes.shape[0]} icd codes remaining'
)
indexes = pl.DataFrame({'icd9_id': range(1, icd9_codes.shape[0]+1)}, schema={'icd9_id':pl.UInt32})
icd9_codes = pl.concat([icd9_codes, indexes], how='horizontal')
diagnoses = (
    diagnoses
    .join (
        icd9_codes,
        on = 'icd_code',
        how = 'left',
    )
    .with_columns(icd9_id=col('icd9_id').fill_null(pl.lit(0)))
    .drop('icd_code')
)

# add time data
admissions = admissions.select (
    col('hadm_id'),
    col('admittime').str.to_datetime('%Y-%m-%d %H:%M:%S')
)
diagnoses = diagnoses.join(admissions, on='hadm_id', how='left').drop('hadm_id')

# prepare for use
diagnoses_a = (
    diagnoses
    .with_columns(
        position = col('admittime').rank('dense').over('subject_id') - 1,
    )
    .group_by('subject_id')
    .agg(
        col(['ccs_id', 'icd9_id', 'position']).sort_by('admittime'),
    )
)
diagnoses_b = (
    diagnoses
    .group_by(['subject_id', 'admittime'])
    .agg(
        pl.count(),
     )
    .group_by('subject_id')
    .agg(col('count').sort_by('admittime'))
)
diagnoses = diagnoses_a.join(diagnoses_b, on='subject_id', how='inner')

# Build Ontology

ontology   = pl.read_csv(ontology_file)

def remove_prefixes(exp: pl.Expr, prefixes: list[str]) -> pl.Expr:
    for p in prefixes:
        exp = exp.str.strip_prefix(p)
    return exp
ontology = ontology.lazy().select(
    label    = pl.col('Preferred Label'),
    icd_code = remove_prefixes(pl.col('Class ID'), ontology_prefixes),
    parent   = remove_prefixes(pl.col('Parents' ), ontology_prefixes),
).collect()


diagnoses_type = ontology.filter(pl.col('parent').str.starts_with('http') & (pl.col('label') != 'PROCEDURES'))['icd_code']

num_rows = 0
while num_rows != len(diagnoses_type):
    num_rows = len(diagnoses_type)
    t = pl.DataFrame({'parent': diagnoses_type})
    t = t.join(ontology[['parent', 'icd_code']], how='left', on='parent')
    diagnoses_type = pl.concat([t['icd_code'], diagnoses_type]).unique()
t = pl.DataFrame({'icd_code': diagnoses_type})
ontology = t.join(ontology, how='left', on='icd_code')

ontology = ontology.with_columns(
    icd_code = pl.col('icd_code').str.replace('.', '', literal=True),
    parent   = pl.when (
        pl.col('parent').str.starts_with('http') 
    )
    .then(pl.lit(ontology_root_name))
    .otherwise(pl.col('parent').str.replace('.', '', literal=True)),
)

ontology = ontology.join(icd9_codes, on='icd_code', how='outer')

uncoded = ontology.filter(pl.col('icd9_id').is_null())
first_new_id = icd9_codes['icd9_id'].max() + 1
indexes = pl.DataFrame({'icd9_id': range(first_new_id, first_new_id+len(uncoded))}, schema={'icd9_id':pl.UInt32})
uncoded = pl.concat([uncoded.drop('icd9_id'), indexes], how='horizontal')

root_id = first_new_id + len(uncoded)
# schema is {'icd_code': Utf8, 'label': Utf8, 'parent': Utf8, 'icd9_id': UInt32}
root_row = pl.DataFrame(
    {'icd_code':ontology_root_name, 'label':'Root of Ontology', 'parent':'root', 'icd9_id':root_id},
    schema = ontology.schema,
)

ontology = pl.concat([
    ontology.filter(~pl.col('icd9_id').is_null()).sort('icd9_id'),
    uncoded,
    root_row,
])

dictionary = ontology.select(parent=pl.col('icd_code'), parent_id=pl.col('icd9_id'))
ontology = ontology.join(dictionary, how='left', on='parent')

# Save

diagnoses_path = os.path.join(output_prefix, output_diagnoses)
ccs_path       = os.path.join(output_prefix, output_ccs)
icd9_path      = os.path.join(output_prefix, output_icd)
ontology_path  = os.path.join(output_prefix, output_ontology)

print(f'diagnoses path is: { diagnoses_path}')
print(f'ccs path is:       { ccs_path}')
print(f'icd9 path is:      { icd9_path}')
print(f'ontology path is:  { ontology_path}')

diagnoses .write_parquet(diagnoses_path)
ccs_codes .write_parquet(ccs_path)
icd9_codes.write_parquet(icd9_path)
ontology  .write_parquet(ontology_path)


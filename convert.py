import polars as pl
from polars import col

import os

mimic_prefix = 'data/mimic-iv/2.2/hosp/'
ccs_prefix   = 'data/ccs'

diagnoses_file  = 'diagnoses_icd.csv.gz'
admissions_file = 'admissions.csv.gz'
ccs_single_file = 'ccs_single_dx_tool_2015.csv'
icd_conv_file   = 'icd10cmtoicd9gem.csv'

diagnoses_path = os.path.join(mimic_prefix, diagnoses_file)
diagnoses      = pl.read_csv(diagnoses_path)
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

diagnoses_icd10 = (
    diagnoses
        .filter(col('icd_version') == 10)
        .join(icd_conv, left_on='icd_code', right_on='icd10cm', how='left')
        .select(col('subject_id'), col('hadm_id'), col('icd9cm').alias('icd_code'))
)
diagnoses_icd9 = diagnoses.filter(col('icd_version') == 9)[diagnoses_icd10.columns]
diagnoses = pl.concat([diagnoses_icd9, diagnoses_icd10], how='vertical')

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
        ccs = col('ccs').fill_null(pl.lit(0)) # @todo this is a lazy way to deal with null values
    )
    .drop('icd_code')
)

ccs_codes = diagnoses[['ccs']].unique().sort('ccs')
we_have_zero_code = 0 in ccs_codes['ccs'].head(1)
if not we_have_zero_code:
    print('WARNING: we do not have a zero code!')
ccs_codes = ccs_codes.join(ccs_conv.unique('ccs'), on='ccs', how='left')
starting_index = 0 if we_have_zero_code else 1
indexes = pl.DataFrame({'code_id': range(starting_index, starting_index+ccs_codes.shape[0])})
ccs_codes = pl.concat([ccs_codes, indexes], how='horizontal')

diagnoses = diagnoses.join(ccs_codes[['ccs', 'code_id']], on='ccs', how='left').drop('ccs')
diagnoses = diagnoses.group_by(col('*').exclude('code_id')).agg(col('code_id').alias('code_ids'))

admissions = admissions.select (
    col('hadm_id'),
    col('admittime').str.to_datetime('%Y-%m-%d %H:%M:%S')
)
diagnoses = diagnoses.join(admissions, on='hadm_id', how='left')

diagnoses = diagnoses.group_by('subject_id').agg(col('code_ids').sort_by('admittime'))

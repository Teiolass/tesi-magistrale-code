import os
from typing import TypeVar
import pandas as pd
import numpy as np

_Mimic = TypeVar('_Mimic', bound='Mimic') # this should be converted to `Self` in python 3.11
class Mimic:
    """
        holds all the informations about the location of mimic and ccs files
    """
    diagnoses : str
    admissions: str
    ccs_convert: str

    DEFAULT_DIAGNOSES  = 'DIAGNOSES_ICD.csv.gz'
    DEFAULT_ADMISSIONS = 'ADMISSIONS.csv.gz'
    DEFAULT_CCS_CONVERT = 'dxicd2ccsxw.csv'

    def __init__(self, *, diagnoses: str, admissions: str, ccs_convert: str):
        self.diagnoses  = diagnoses
        self.admissions = admissions
        self.ccs_convert = ccs_convert

    def from_folder(folder: str, ccs_convert: str) -> _Mimic:
        """
            returns a Mimic object from given folder with default file names
        """
        diagnoses  = os.path.join(folder, Mimic.DEFAULT_DIAGNOSES)
        admissions = os.path.join(folder, Mimic.DEFAULT_ADMISSIONS)
        ccs_convert = os.path.join(ccs_convert, Mimic.DEFAULT_CCS_CONVERT)
        return Mimic (
            diagnoses  = diagnoses,
            admissions = admissions,
            ccs_convert = ccs_convert,
        )



_MimicData = TypeVar('_MimicData', bound='MimicData')  # this should be converted to `Self` in python 3.11
class MimicData:
    codes     : np.ndarray
    visits    : np.ndarray
    patients  : np.ndarray
    encodings : np.ndarray

    def __init__(self, *, codes, visits, patients, encodings):
        self.codes     = codes
        self.visits    = visits
        self.patients  = patients
        self.encodings = encodings
    
    def from_mimic(mimic: Mimic) -> _MimicData:
        diagnoses  = pd.read_csv(mimic.diagnoses)
        admissions = pd.read_csv(mimic.admissions)
        ccs_convert = pd.read_csv(mimic.ccs_convert)

        # remove null diagnoses in diagnoses
        num_null = diagnoses.ICD9_CODE.isna().sum()
        if num_null > 0:
            print(f'INFO: removing {num_null} invalid records from diagnoses')
            diagnoses = diagnoses.dropna(subset=['ICD9_CODE'])
        del num_null

        # remove non-referenced admission entries
        ref_mask = admissions.HADM_ID.isin(diagnoses.HADM_ID)
        num_unref = (~ref_mask).sum()
        if num_unref > 0:
            print(f'INFO: dropping {num_unref} unreferenced records from admissions')
            admissions = admissions[ref_mask]
        del ref_mask, num_unref

        # filter patients with at least two visits
        eligible = admissions.SUBJECT_ID.value_counts()
        eligible = eligible[eligible > 1]
        eligible_mask = admissions.SUBJECT_ID.isin(eligible.index)
        num_uneligible = (~eligible_mask).sum()
        if num_uneligible > 0:
            print(f'INFO: dropping {num_uneligible} records from admissions of patients with less than two visits')
            admissions = admissions[eligible_mask]
            diagnoses_mask = diagnoses.SUBJECT_ID.isin(eligible.index)
            num_removing = (~diagnoses_mask).sum()
            print(f'INFO: dropping {num_removing} records from diagnoses of patients with less than two visits')
            diagnoses = diagnoses[diagnoses_mask]
        del eligible, eligible_mask, num_uneligible, num_removing, diagnoses_mask

        # do the conversion for codes
        # @todo this will be include the ccs conversion!
        raw_codes = pd.merge(
            diagnoses.ICD9_CODE,
            ccs_convert[['icd', 'ccs']],
            how = 'left',
            left_on = 'ICD9_CODE',
            right_on = 'icd',
        )
        cat_codes = raw_codes.ccs.astype('category')
        codes = cat_codes.cat.codes
        encodings = cat_codes.cat.categories
        del raw_codes, cat_codes

        # augment diagnoses info with time
        admissions['int_time'] = admissions.ADMITTIME.apply(pd.to_datetime).astype(int)
        diagnoses = diagnoses.merge(admissions[['HADM_ID', 'int_time']], on='HADM_ID', how='left')
        del admissions

        # sort diagnoses. Is it really useful?
        diagnoses.sort_values(by=['SUBJECT_ID', 'int_time'])

        # find lengths of visits and patients histories
        diagnoses = diagnoses[['SUBJECT_ID', 'int_time']]
        visits = diagnoses.groupby(['SUBJECT_ID', 'int_time'], sort=False).size().sort_index().reset_index()
        patients = visits.groupby('SUBJECT_ID', sort=False).int_time.count().sort_index().reset_index()
        visits = visits.drop(columns='SUBJECT_ID')

        return MimicData (
            codes     = codes.to_numpy(),
            visits    = visits.to_numpy(),
            patients  = patients.to_numpy(),
            encodings = encodings.to_numpy(),
        )


if __name__ == '__main__':
    mimic = Mimic.from_folder('/home/amarchetti/mimic-iii', '/home/amarchetti')
    data = MimicData.from_mimic(mimic)


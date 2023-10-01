import os

import pandas as pd
import numpy as np
from typing import Self
from dataclasses import dataclass

RANDOM_STATE = 42 # reproducibility

@dataclass(kw_only=True)
class Mimic:
    """
        holds all the informations about the location of mimic and ccs files
    """
    diagnoses  : str
    admissions : str
    ccs_convert: str

    DEFAULT_DIAGNOSES   = 'DIAGNOSES_ICD.csv'
    DEFAULT_ADMISSIONS  = 'ADMISSIONS.csv'
    DEFAULT_CCS_CONVERT = 'dxicd2ccsxw.csv'

    def from_folder(folder: str, ccs_convert: str) -> Self:
        """
            returns a Mimic object from given folder with default file names
        """
        diagnoses   = os.path.join(folder,      Mimic.DEFAULT_DIAGNOSES)
        admissions  = os.path.join(folder,      Mimic.DEFAULT_ADMISSIONS)
        ccs_convert = os.path.join(ccs_convert, Mimic.DEFAULT_CCS_CONVERT)
        return Mimic (
            diagnoses   = diagnoses,
            admissions  = admissions,
            ccs_convert = ccs_convert,
        )



@dataclass(kw_only=True)
class MimicData:
    codes     : np.ndarray
    visits    : np.ndarray
    patients  : np.ndarray
    encodings : np.ndarray

    def from_mimic(mimic: Mimic, *, pad_visits=False, pad_codes=False) -> Self:
        """
            Load and preprocess data from raw mimic files
            WARNING: codes and encodings are shifted by 1 (code 0 means empty or invalid)
        """
        # load only the columns we are interest to
        diagnoses_cols  = ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']
        admissions_cols = ['HADM_ID', 'ADMITTIME', 'SUBJECT_ID']
        ccs_cols        = ['icd', 'ccs'] 
        diagnoses   = pd.read_csv(mimic.diagnoses,   usecols=diagnoses_cols)
        admissions  = pd.read_csv(mimic.admissions,  usecols=admissions_cols)
        ccs_convert = pd.read_csv(mimic.ccs_convert, usecols=ccs_cols)

        # remove null diagnoses in diagnoses
        num_null = diagnoses.ICD9_CODE.isna().sum()
        if num_null > 0:
            print(f'INFO: removing {num_null} null records from diagnoses')
            diagnoses = diagnoses.dropna(subset=['ICD9_CODE'])
        del num_null

        # remove non-referenced admission entries
        ref_mask  = admissions.HADM_ID.isin(diagnoses.HADM_ID)
        num_unref = (~ref_mask).sum()
        if num_unref > 0:
            print(f'INFO: dropping {num_unref} unreferenced records from admissions')
            admissions = admissions[ref_mask]
        del ref_mask, num_unref

        # filter patients with strictly more than a single visit
        eligible = admissions.SUBJECT_ID.value_counts()
        eligible = eligible[eligible > 1]
        eligible_mask  = admissions.SUBJECT_ID.isin(eligible.index)
        num_uneligible = (~eligible_mask).sum()
        if num_uneligible > 0:
            print(f'INFO: dropping {num_uneligible} records from admissions of patients with less than two visits')
            admissions     = admissions[eligible_mask]
            diagnoses_mask = diagnoses.SUBJECT_ID.isin(eligible.index)
            num_removing   = (~diagnoses_mask).sum()
            print(f'INFO: dropping {num_removing} records from diagnoses of patients with less than two visits')
            diagnoses = diagnoses[diagnoses_mask]
        # we will use eligble later to shuffle the patients
        del eligible_mask, num_uneligible, num_removing, diagnoses_mask

        # do the conversion for codes
        ccs_convert = ccs_convert.set_index('icd').ccs
        raw_codes = diagnoses.ICD9_CODE.map(ccs_convert)
        cat_codes = raw_codes.astype('category')
        codes     = cat_codes.cat.codes + 1  
        encodings = cat_codes.cat.categories
        del raw_codes, cat_codes

        # augment diagnoses infos with time
        admissions['int_time'] = admissions.ADMITTIME.apply(pd.to_datetime).astype(int)
        diagnoses = diagnoses.merge(admissions[['HADM_ID', 'int_time']], on='HADM_ID', how='left')

        # shuffle patients
        unique_patients = pd.Series(eligible.index)
        unique_patients = unique_patients.sample(frac=1.0, random_state=RANDOM_STATE)
        unique_patients = unique_patients.reset_index(drop=True)
        unique_patients = pd.Series(np.arange(unique_patients.size), index=unique_patients, name='tmp')
        diagnoses = diagnoses.join(unique_patients, on='SUBJECT_ID', how='left')
        diagnoses = diagnoses.drop(columns='SUBJECT_ID').rename(columns={'tmp': 'SUBJECT_ID'})
        del unique_patients
        diagnoses.sort_values(by=['SUBJECT_ID', 'int_time'])

        # find lengths of visits and patients histories
        diagnoses = diagnoses[['SUBJECT_ID', 'int_time']]
        visits    = diagnoses.groupby(['SUBJECT_ID', 'int_time'], sort=False).size().sort_index().reset_index()
        patients  = visits.groupby('SUBJECT_ID', sort=False).int_time.count().sort_index().reset_index()
        visits    = visits.drop(columns='SUBJECT_ID')

        # convert to numpy and pad what must be padded
        p_buffer = np.empty((len(patients) + 1), dtype=np.int32)
        patients = patients.int_time.to_numpy()
        p_buffer[1:] = patients
        p_buffer[0]  = 0
        np.cumsum(p_buffer, out=p_buffer)

        visits = visits[0].to_numpy(dtype=np.int32)

        # sketchy things to pad all the vectors
        codes = codes.to_numpy(dtype=np.int32)
        if pad_codes:
            SIZE = 64
            num_pads = SIZE - visits
            pad = np.zeros(num_pads.sum(), dtype=codes.dtype)
            pad_pos = np.repeat(np.cumsum(visits), num_pads)
            codes = np.insert(codes, pad_pos, pad)
            codes = codes.reshape((-1, SIZE))

        if pad_visits:
            SIZE = 64
            num_pads = SIZE - patients
            pad = np.zeros(num_pads.sum(), dtype=visits.dtype)
            pad_pos = np.repeat(p_buffer[1:], num_pads)
            visits = np.insert(visits, pad_pos, pad)
            visits = visits.reshape((-1, SIZE))
        else:
            buffer = np.empty((len(visits) + 1), dtype=np.int32)
            buffer[1:] = visits
            buffer[0] = 0
            visits = buffer
            np.cumsum(visits, out=visits)

        patients = p_buffer

        return MimicData (
            codes     = codes,
            visits    = visits,
            patients  = patients,
            encodings = encodings.to_numpy(),
        )

    def get_num_codes(self) -> int:
        return self.encodings.size + 1

if __name__ == '__main__':
    # these paths are relevant for the cnr machine only
    if not 'mimic' in globals() or not 'data' in globals():
        print('it seems that the variables `mimic` and `data` are not defined in the global namespace')
        print('I`m going to create them')
        mimic = Mimic.from_folder('/home/amarchetti/data/mimic-iii', '/home/amarchetti/data')
        data  = MimicData.from_mimic(mimic, pad_visits=False, pad_codes=True)
        print('data loaded')
    else:
        print('I have found the variables `mimic` and `data` in the global namespace')
        print('I think there is no need to recompute them')
    print()

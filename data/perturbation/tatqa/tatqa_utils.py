# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import json
from pathlib import Path
from typing import Dict


def copy_over(split, original, target):
    assert split in ['train', 'valid', 'test']
    work_path = Path('dataset')
    original_content = (work_path / Path(original) / Path(split + 'set').with_suffix('.json')).open().read()
    (work_path / Path(target) / Path(split + 'set').with_suffix('.json')).open('w').write(original_content)
    print('Copied from', (work_path / Path(original) / Path(split + 'set').with_suffix('.json')))
    print('Copied to', (work_path / Path(target) / Path(split + 'set').with_suffix('.json')))


def create_atk_da_def(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                      dataset_name: str, attack_name: str, convert_f,
                      do_attack=True, do_DA=True, do_defense=True, dump_json=True):

    print('Building', attack_name)
    ret = {}

    if do_attack:
        # Attack
        print('Start Attacking')
        dataset_path = Path(f'dataset/{dataset_name}/')
        pert_test = []
        rest_test = []
        for idx, data_row in test_df.iterrows():
            if convert_f(data_row) is not None:
                pert_test.append(convert_f(data_row))
            else:
                rest_test.append(data_row.to_dict())
        attack_test = pert_test + rest_test
        print(f'attack test: {len(pert_test)}, rest: {len(rest_test)}, original: {len(test_df)}')
        if len(attack_test) < len(test_df):
            print('Warning: Some samples filtered but not perturbed')
        ret.update({'pert_test': pert_test, 'rest_test': rest_test})

        if dump_json:
            # create the directory
            setting_suffix = 'atk'
            dump_path = dataset_path.parent / Path(f'{dataset_name}-{attack_name}-{setting_suffix}')
            dump_path.mkdir(parents=True, exist_ok=True)
            with open(dump_path / Path('testset.json'), 'w') as f:
                json.dump(attack_test, f)

            # pack a full dataset
            copy_over('train', dataset_name, f'{dataset_name}-{attack_name}-{setting_suffix}')
            copy_over('valid', dataset_name, f'{dataset_name}-{attack_name}-{setting_suffix}')
            print('train and valid set copied over, Attack finished.')

    if do_DA:
        # TODO: improve the efficiency here. Can do df.map, believed to be important
        # Data Augmenting
        print('Start Data Augmenting')
        pert_train = [convert_f(data) for _, data in train_df.iterrows() if convert_f(data) is not None]
        defense_train = pert_train + train_df.to_dict('records')
        pert_valid = [convert_f(data) for _, data in valid_df.iterrows() if convert_f(data) is not None]
        defense_valid = pert_valid + valid_df.to_dict('records')
        print(f'defense train: {len(pert_train)} new + {len(train_df)} original')
        print(f'defense valid: {len(pert_valid)} new + {len(valid_df)} original')
        ret.update({'pert_train': pert_train, 'pert_valid': pert_valid})

        if dump_json:
            # create the directory
            setting_suffix = 'da'
            dump_path = dataset_path.parent / Path(f'{dataset_name}-{attack_name}-{setting_suffix}')
            dump_path.mkdir(parents=True, exist_ok=True)
            with open(dump_path / Path('trainset.json'), 'w') as f:
                json.dump(defense_train, f)
            with open(dump_path / Path('validset.json'), 'w') as f:
                json.dump(defense_valid, f)

            # pack a full dataset
            copy_over('test', dataset_name, f'{dataset_name}-{attack_name}-{setting_suffix}')
            print('test set copied over, DA finished')

    if do_defense:
        # Defense
        print('Start Defense')

        if dump_json:
            # create the directory
            setting_suffix = 'def'
            dump_path = dataset_path.parent / Path(f'{dataset_name}-{attack_name}-{setting_suffix}')
            dump_path.mkdir(parents=True, exist_ok=True)

            # pack a full dataset
            copy_over('train', f'{dataset_name}-{attack_name}-da', f'{dataset_name}-{attack_name}-{setting_suffix}')
            copy_over('valid', f'{dataset_name}-{attack_name}-da', f'{dataset_name}-{attack_name}-{setting_suffix}')
            copy_over('test', f'{dataset_name}-{attack_name}-atk', f'{dataset_name}-{attack_name}-{setting_suffix}')
            print('datasets copied over, Defense finished')

    print('Finished', attack_name)
    print('\n\n')
    return ret


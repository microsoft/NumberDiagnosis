# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import json
from typing import Dict


def batch_property_change(root_dataset_name: str, properties: Dict):
    assert root_dataset_name in ['asdiv-a', 'mawps']
    work_path = Path.cwd().parent/Path('mwptoolkit/properties/dataset')
    # print(work_path)
    for p in work_path.glob(f'{root_dataset_name}*.json'):
        with p.open() as f:
            content = json.load(f)
        content.update(properties)
        with p.open('w') as f:
            json.dump(content, f)


def create_property(dataset_name, original_name=None):
    work_path = Path('mwptoolkit/properties/dataset')
    if original_name is None:
        original_name = '-'.join(dataset_name.split('-')[:-2])
        print('inferred original name:', original_name)

    # read original content
    original_path = work_path/Path(original_name).with_suffix('.json')
    with original_path.open() as f:
        original_content = f.read()

    # write to new propety json file
    new_path = work_path/Path(dataset_name).with_suffix('.json')
    with new_path.open('w') as f:
        f.write(original_content)
    print('Created property at', work_path/Path(dataset_name).with_suffix('.json'))


def copy_over(split, original, target):
    assert split in ['train', 'valid', 'test']
    work_path = Path('dataset')
    original_content = (work_path / Path(original) / Path(split + 'set').with_suffix('.json')).open().read()
    (work_path / Path(target) / Path(split + 'set').with_suffix('.json')).open('w').write(original_content)
    print('Copied from', (work_path / Path(original) / Path(split + 'set').with_suffix('.json')))
    print('Copied to', (work_path / Path(target) / Path(split + 'set').with_suffix('.json')))


def create_atk_da_def(train, valid, test, dataset_name, attack_name, convert_f,
                      filter_f=lambda x: True, do_attack=True, do_DA=True, do_defense=True,
                      dump_json=True):

    print('Building', attack_name)
    ret = {}
    dataset_path = Path(f'dataset/{dataset_name}/')

    if do_attack:
        # Attack
        print('Start Attacking')
        pert_test = []
        rest_test = []
        for _ in test:
            if filter_f(_) and convert_f(_) is not None:
                pert_test.append(convert_f(_))
            else:
                rest_test.append(_)
        attack_test = pert_test + rest_test
        print(f'attack test: {len(pert_test)}, rest: {len(rest_test)}, original: {len(test)}')
        if len(attack_test) < len(test):
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
            create_property(f'{dataset_name}-{attack_name}-{setting_suffix}')
            print('train and valid set copied over, Attack finished.')

    if do_DA:
        # Data Augmenting
        print('Start Data Augmenting')
        pert_train = [convert_f(_) for _ in train if filter_f(_) and convert_f(_) is not None]
        defense_train = pert_train + train
        pert_valid = [convert_f(_) for _ in valid if filter_f(_) and convert_f(_) is not None]
        defense_valid = pert_valid + valid
        print(f'defense train: {len(pert_train)} new + {len(train)} original')
        print(f'defense valid: {len(pert_valid)} new + {len(valid)} original')
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
            create_property(f'{dataset_name}-{attack_name}-{setting_suffix}', dataset_name)
            print('test set copied over, DA finished')

    if do_defense:
        # Defense
        print('Start Defense')

        defense_train = [convert_f(_) if filter_f(_) and convert_f(_) is not None else _ for _ in train ]
        defense_valid = [convert_f(_) if filter_f(_) and convert_f(_) is not None else _ for _ in valid ]
        
        if dump_json:
            # create the directory
            setting_suffix = 'def'
            dump_path = dataset_path.parent / Path(f'{dataset_name}-{attack_name}-{setting_suffix}')
            dump_path.mkdir(parents=True, exist_ok=True)
            with open(dump_path / Path('trainset.json'), 'w') as f:
                json.dump(defense_train, f)
            with open(dump_path / Path('validset.json'), 'w') as f:
                json.dump(defense_valid, f)

            # pack a full dataset
            copy_over('test', f'{dataset_name}-{attack_name}-atk', f'{dataset_name}-{attack_name}-{setting_suffix}')
            create_property(f'{dataset_name}-{attack_name}-{setting_suffix}', dataset_name)
            print('datasets copied over, Defense finished')

    print('Finished', attack_name)
    print('\n\n')
    return ret


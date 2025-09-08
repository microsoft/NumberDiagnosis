# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path
from perturbation.utils import create_atk_da_def
from .asdiv_a import is_simple, is_simple_for_distri
from .asdiv_a import convert_type, convert_distri, convert_lang, convert_verbosity, convert_noise


dataset_path = Path('dataset/asdiv-a/')

with open(dataset_path / Path('trainset.json'), 'r') as f:
    train = json.load(f)
with open(dataset_path / Path('testset.json'), 'r') as f:
    test = json.load(f)
with open(dataset_path / Path('validset.json'), 'r') as f:
    valid = json.load(f)
print('train / valid / test numbers:')
print(len(train), len(valid), len(test))

create_atk_da_def(train, valid, test, 'asdiv-a', 'type', convert_f=convert_type, filter_f=is_simple, do_attack=False, do_DA=False, do_defense=True)
create_atk_da_def(train, valid, test, 'asdiv-a', 'noise', convert_f=convert_noise, filter_f=is_simple, do_attack=False, do_DA=False, do_defense=True)
create_atk_da_def(train, valid, test, 'asdiv-a', 'distri', convert_f=convert_distri, filter_f=is_simple, do_attack=False, do_DA=False, do_defense=True)
create_atk_da_def(train, valid, test, 'asdiv-a', 'verbosity', convert_f=convert_verbosity, do_attack=False, do_DA=False, do_defense=True)
create_atk_da_def(train, valid, test, 'asdiv-a', 'lang', convert_f=convert_lang, do_attack=False, do_DA=False, do_defense=True)


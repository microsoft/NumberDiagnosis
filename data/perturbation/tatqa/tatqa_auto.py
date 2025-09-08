# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from .tatqa_utils import create_atk_da_def
from .tatqa import convert_type, convert_lang, convert_verbosity


dataset_name = 'tatqa'
train_df = pd.read_json(f'dataset/{dataset_name}/trainset.json')
valid_df = pd.read_json(f'dataset/{dataset_name}/validset.json')
test_df = pd.read_json(f'dataset/{dataset_name}/testset.json')

print('train / valid / test numbers:')
print(len(train_df), len(valid_df), len(test_df))

create_atk_da_def(train_df, valid_df, test_df, 'tatqa', 'type', convert_f=convert_type, do_defense=False, do_DA=False)
create_atk_da_def(train_df, valid_df, test_df, 'tatqa', 'verbosity', convert_f=convert_verbosity, do_defense=False, do_DA=False)
create_atk_da_def(train_df, valid_df, test_df, 'tatqa', 'lang', convert_f=convert_lang, do_defense=False, do_DA=False)


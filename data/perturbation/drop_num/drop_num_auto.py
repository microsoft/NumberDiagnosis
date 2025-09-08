# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from .drop_utils import create_atk_da_def
from .drop_num import convert_type, convert_lang, convert_verbosity


dataset_name = 'drop-num'
train_df = pd.read_json(f'dataset/{dataset_name}/trainset.json')
valid_df = pd.read_json(f'dataset/{dataset_name}/validset.json')
test_df = pd.read_json(f'dataset/{dataset_name}/testset.json')

print('train / valid / test numbers:')
print(len(train_df), len(valid_df), len(test_df))

create_atk_da_def(train_df, valid_df, test_df, 'drop-num', 'type', convert_f=convert_type)
# create_atk_da_def(train_df, valid_df, test_df, 'asdiv-a', 'noise', convert_f=convert_noise, filter_f=is_simple)
# create_atk_da_def(train_df, valid_df, test_df, 'asdiv-a', 'distri', convert_f=convert_distri, filter_f=is_simple)
create_atk_da_def(train_df, valid_df, test_df, 'drop-num', 'verbosity', convert_f=convert_verbosity)
create_atk_da_def(train_df, valid_df, test_df, 'drop-num', 'lang', convert_f=convert_lang)


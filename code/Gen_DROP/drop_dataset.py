# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, T5Tokenizer
from typing import Union


class DROPDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Union[T5Tokenizer, BartTokenizer],
        source_max_token_len: int = 512,
        target_max_token_len: int = 16,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        source_encoding = self.tokenizer(
            data_row['question'],
            data_row['passage'],
            max_length=self.source_max_token_len,
            padding='max_length',
            truncation='only_second',
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            data_row['answer'],
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        labels = torch.clone(target_encoding['input_ids'])
        if type(self.tokenizer) == T5Tokenizer:
            labels[labels == 0] = -100

        return dict(
            question=data_row['question'],
            body=data_row['passage'],
            answer_text=data_row['answer'],
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            output_ids=target_encoding['input_ids']
        )


class DROPDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: Union[T5Tokenizer, BartTokenizer],
        batch_size: int = 16,
        source_max_token_len: int = 512,
        target_max_token_len: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage=None):
        self.train_dataset = DROPDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.valid_dataset = DROPDataset(
            self.valid_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = DROPDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=8,
            persistent_workers=True
        )

    @staticmethod
    def extract_number_qas(path):
        with open(path) as f:
            data = json.load(f)
        data_rows = []
        for k in list(data.keys()):
            passage = data[k]['passage']
            qa_pairs = data[k]['qa_pairs']
            for qa in qa_pairs:
                if qa['answer']['number'] != '':
                    data_rows.append({
                        'id': k,
                        'passage': passage,
                        'question': qa['question'],
                        'answer': qa['answer']['number']
                    })
        return pd.DataFrame(data_rows)
    

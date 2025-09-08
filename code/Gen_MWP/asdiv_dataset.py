# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, T5Tokenizer
from typing import Union


class ASDivDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Union[T5Tokenizer, BartTokenizer],
        source_max_token_len: int = 512,
        target_max_token_len: int = 32,
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
            data_row['Question'],
            data_row['Body'],
            max_length=self.source_max_token_len,
            padding='max_length',
            truncation='only_second',
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            data_row['Formula'].split('=')[0],
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        labels = target_encoding['input_ids']
        if type(self.tokenizer) == T5Tokenizer:
            labels[labels == 0] = -100

        return dict(
            question=data_row['Question'],
            body=data_row['Body'],
            answer_text=data_row['Formula'].split('=')[0],
            q_id=data_row['@ID'],
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )


class ASDivDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: Union[T5Tokenizer, BartTokenizer],
        batch_size: int = 32,
        source_max_token_len: int = 128,
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
        self.train_dataset = ASDivDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.valid_dataset = ASDivDataset(
            self.valid_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = ASDivDataset(
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
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=8
        )


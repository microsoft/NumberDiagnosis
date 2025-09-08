# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
from .asdiv_dataset import ASDivDataModule
from .asdiv_model import ASDivModel
from .config import GenMWPConfig


def build_trainer(config):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{config.save_dir}',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_ans_acc',
        mode='max'
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=5)],
        max_epochs=config.max_epoch,
        gpus=1  # on the asdiv-a dataset we only use one GPU #TODO: scale up possible? 
    )
    return trainer


def clean_save_dir(config):
    save_dir = Path('checkpoints', config.save_dir)
    if save_dir.exists():
        for ckpt in save_dir.glob('*.ckpt'):
            ckpt.unlink()


def train(config):
    # train on original dataset
    train_df = pd.read_json(f'dataset/{config.dataset_name}/trainset.json').astype(str)
    valid_df = pd.read_json(f'dataset/{config.dataset_name}/validset.json').astype(str)
    test_df = pd.read_json(f'dataset/{config.dataset_name}/testset.json').astype(str)

    if config.model_name == 'bart':
        pretrained_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=config.model_max_length)
    elif config.model_name == 't5':
        pretrained_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=config.model_max_length)
    else:
        raise Exception('Model name not "bart" or "t5"')

    model = ASDivModel(pretrained_model, tokenizer)
    model.test_len, model.valid_len = len(test_df), len(valid_df)
    model.result_id = f'{config.model_name}-{config.dataset_name}'
    data_module = ASDivDataModule(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        tokenizer=tokenizer,
        source_max_token_len=config.model_max_length
    )
    data_module.setup()
    trainer = build_trainer(config)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())


def test(config):
    print(f'Reading from dataset/{config.dataset_name}')
    train_df = pd.read_json(f'dataset/{config.dataset_name}/trainset.json').astype(str)
    valid_df = pd.read_json(f'dataset/{config.dataset_name}/validset.json').astype(str)
    test_df = pd.read_json(f'dataset/{config.dataset_name}/testset.json').astype(str)
    
    check_point_path = f'checkpoints/{config.load_dir}/best-checkpoint.ckpt'
    if config.model_name == 'bart':
        pretrained_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=config.model_max_length)
    elif config.model_name == 't5':
        pretrained_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=config.model_max_length)
    else:
        raise Exception('Model name not bart')
    model = ASDivModel.load_from_checkpoint(check_point_path,
                                            model=pretrained_model,
                                            tokenizer=tokenizer)
    model.test_len, model.valid_len = len(test_df), len(valid_df)
    model.result_id = f'{config.model_name}-{config.dataset_name}'
    
    trainer = build_trainer(config)  # don't really need to train
    data_module = ASDivDataModule(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        tokenizer=tokenizer,
        source_max_token_len=config.model_max_length
    )
    data_module.setup()
    
    trainer.test(model, data_module.test_dataloader())


def main(args):
    pl.seed_everything(args.seed)
    config = GenMWPConfig(args)

    if config.setting_name == 'atk':
        # Train on original, test on perturbed
        if not args.test_only:
            config.setup(perturb='original')
            clean_save_dir(config)
            train(config)

        # for perturb in ['original', 'type', 'noise', 'distri', 'verbosity', 'lang']:
        for perturb in args.perturbs:
            config.setup_test(perturb=perturb, train_perturb='original')
            test(config)

    elif config.setting_name == 'def':
        # Train on perturbed
        for perturb in args.perturbs:
            if not args.test_only:
                config.setup(perturb=perturb)
                clean_save_dir(config)
                train(config)
            config.setup_test(perturb=perturb, train_perturb=perturb)
            test(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # model args
    parser.add_argument('--model_name', type=str, default='bart', help='name of models')
    parser.add_argument('--model_max_length', type=int, default=128, help='model length for BART on ASDiv-a, '
                                                                          'subject to change when the dataset is larger')
    # dataset args
    parser.add_argument('--root_dataset_name', type=str, default='asdiv-a', help='name of datasets')
    parser.add_argument('--perturbs', nargs='+')
    # training process args
    parser.add_argument('--setting_name', type=str, default='atk', help='name of current setting', choices=['atk', 'def', 'da'])
    parser.add_argument('--max_epoch', type=int, default=80, help='the number of maximum training epochs')
    parser.add_argument('--test_only', action='store_true')
    
    args = parser.parse_args()
    main(args)


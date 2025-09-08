# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
from .drop_dataset import DROPDataModule
from .drop_model import DROPModel
from .config import GenDROPConfig


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
        callbacks=[checkpoint_callback, pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)],
        strategy="ddp_find_unused_parameters_false",
        max_epochs=config.max_epochs,
        gpus=config.gpu_num
    )
    return trainer


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
        raise Exception('Model name not bart')

    model = DROPModel(pretrained_model, tokenizer)
    model.test_len, model.valid_len = len(test_df), len(valid_df)
    data_module = DROPDataModule(
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
    model = DROPModel.load_from_checkpoint(check_point_path,
                                            model=pretrained_model,
                                            tokenizer=tokenizer)
    model.test_len, model.valid_len = len(test_df), len(valid_df)
    
    trainer = build_trainer(config)  # don't really need to train
    data_module = DROPDataModule(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        tokenizer=tokenizer,
        source_max_token_len=config.model_max_length,
        batch_size=config.batch_size
    )
    data_module.setup()
    trainer.test(model, data_module.test_dataloader())


def main(args):
    pl.seed_everything(args.seed)
    config = GenDROPConfig(args)

    if config.setting_name == 'atk':
        # Train on original, test on perturbed
        if args.train_test == 'train':
            config.setup_train(perturb='original')
            train(config)
        elif args.train_test == 'test':
            for perturb in args.perturbs:
                config.setup_test(perturb=perturb, trained_perturb='original')
                test(config)
        else:
            raise Exception('train/test unclear!')

    elif config.setting_name == 'def':
        # Train on perturbed
        for perturb in args.perturbs:
            if args.train_test == 'train':
                config.setup_train(perturb=perturb)
                train(config)
            elif args.train_test == 'test':
                config.setup_test(perturb=perturb, trained_perturb=perturb)
                test(config)
            else:
                raise Exception('train/test unclear!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # model args
    parser.add_argument('--model_name', type=str, default='bart', help='name of models')
    parser.add_argument('--model_max_length', type=int, default=384, help='model length for BART on DROP, '
                                                                          'subject to change when the dataset is larger')
    # dataset args
    parser.add_argument('--root_dataset_name', type=str, default='drop-num', help='name of datasets')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size used in dataset')
    parser.add_argument('--perturbs', nargs='+')
    # training process args
    parser.add_argument('--setting_name', type=str, default='atk', help='name of current setting', choices=['atk', 'def', 'da'])
    parser.add_argument('--max_epochs', type=int, default=20, help='the number of maximum training epochs')
    parser.add_argument('--train_test', type=str, help='Whether the process is for training or testing', choices=['train', 'test'])
    
    args = parser.parse_args()
    main(args)


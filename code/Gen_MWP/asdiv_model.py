# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import pytorch_lightning as pl
import json
from pathlib import Path


class ASDivModel(pl.LightningModule):

    def __init__(self, model=None, tokenizer=None, valid_len=1, test_len=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.test_len = test_len
        self.valid_len = valid_len
        self.result_id = None

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        # return output.loss, output.logits
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask, labels)
        loss, outputs = output.loss, output.logits
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        answer_text = batch['answer_text'][0]
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if pred == answer_text:
            eq_correct = True
            ans_correct = True
        else:
            eq_correct = False
            try:
                if abs(eval(pred) - eval(answer_text)) < 1e-4:
                    ans_correct = True
                else:
                    ans_correct = False
            except (SyntaxError, TypeError, NameError, ZeroDivisionError):
                ans_correct = False
        return eq_correct, ans_correct

    def validation_epoch_end(self, validation_step_outputs):
        eq_cnt = [_[0] for _ in validation_step_outputs].count(True)
        ans_cnt = [_[1] for _ in validation_step_outputs].count(True)
        self.log('val_eq_acc', eq_cnt/self.valid_len, prog_bar=True, logger=True)
        self.log('val_ans_acc', ans_cnt/self.valid_len, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        answer_text = batch['answer_text'][0]
        q_id = batch['q_id'][0]
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # if batch_idx < 5:
        #     print(f'pred: "{pred}", ground truth: "{answer_text}"')
        if pred == answer_text:
            eq_correct = True
            ans_correct = True
        else:
            eq_correct = False
            try:
                if abs(eval(pred) - eval(answer_text)) < 1e-4:
                    ans_correct = True
                else:
                    ans_correct = False
            except (SyntaxError, TypeError, NameError, ZeroDivisionError):
                ans_correct = False
        result = {
            'id': q_id,
            'prediction': pred,
            'target': answer_text
        }
        return eq_correct, ans_correct, result

    def test_epoch_end(self, test_step_outputs):
        eq_cnt = [_[0] for _ in test_step_outputs].count(True)
        ans_cnt = [_[1] for _ in test_step_outputs].count(True)
        results = [_[2] for _ in test_step_outputs]
        self.log({'eq_acc': eq_cnt/self.test_len})
        self.log({'ans_acc': ans_cnt/self.test_len})
        save_path = Path('result', self.result_id)
        save_path.mkdir(exist_ok=True, parents=True)
        with open(save_path/'generation_result.json', 'w') as f:
            json.dump(results, f)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


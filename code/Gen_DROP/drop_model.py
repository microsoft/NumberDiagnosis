# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import pytorch_lightning as pl
from transformers import BartTokenizer, T5Tokenizer

class DROPModel(pl.LightningModule):

    def __init__(self, model=None, tokenizer=None, valid_len=1, test_len=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.test_len = test_len
        self.valid_len = valid_len

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
        output_ids = batch['output_ids']
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output_ids = output_ids.squeeze(1)
        if type(self.tokenizer) == T5Tokenizer:
            pred_ids = torch.zeros_like(output_ids, device=self.device)
        elif type(self.tokenizer) == BartTokenizer:
            pred_ids = torch.ones_like(output_ids, device=self.device)
        else:
            raise Exception('Tokenizer type unkown during validation')
        pred_ids[:, :generated_ids.shape[1]-1] = generated_ids[:, 1:]  # pad the generated ids
        ans_correct = torch.sum((pred_ids == output_ids).all(dim=1))
        return ans_correct

    def validation_epoch_end(self, validation_step_outputs):
        ans_cnt = torch.sum(torch.tensor(validation_step_outputs))
        self.log('val_ans_acc', ans_cnt/self.valid_len, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        answer_text = batch['answer_text'][0]
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if pred == answer_text:
            ans_correct = True
        else:
            try:
                if abs(eval(pred) - eval(answer_text)) < 1e-4:
                    ans_correct = True
                else:
                    ans_correct = False
            except (SyntaxError, TypeError, NameError, ZeroDivisionError):
                ans_correct = False
        return ans_correct

    def test_epoch_end(self, test_step_outputs):
        ans_cnt = torch.sum(torch.tensor(test_step_outputs))
        self.log('test_ans_acc', ans_cnt/self.test_len, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


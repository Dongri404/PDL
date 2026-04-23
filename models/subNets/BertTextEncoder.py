import os
import pdb
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel, AutoModel, AutoTokenizer

__all__ = ['RobertaTextEncoder']
    
class RobertaTextEncoder(nn.Module):
    def __init__(self, language='en',raw_text=False, use_finetune=False):
        super().__init__()
        if language == 'en':
            self.max_length = 50
            self.tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/zwj/PretrainedModels/FacebookAI/roberta-large")
            self.roberta_model = RobertaModel.from_pretrained('/mnt/disk2/zwj/PretrainedModels/FacebookAI/roberta-large')
        elif language == 'cn':
            self.max_length = 39
            self.tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/zwj/PretrainedModels/hfl/chinese-roberta-wwm-ext")
            self.roberta_model = AutoModel.from_pretrained("/mnt/disk2/zwj/PretrainedModels/hfl/chinese-roberta-wwm-ext")
        self.use_finetune = use_finetune
        self.raw_text = raw_text

    def forward(self,text_inputs):
        if self.raw_text:
            tokenized = self.tokenizer.batch_encode_plus(
            text_inputs,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
            )
            input_ids, input_mask = tokenized['input_ids'].cuda(), tokenized['attention_mask'].cuda()
        else:
            input_ids, input_mask, _ = text_inputs[:,0,:].long().cuda(), text_inputs[:,1,:].float().cuda(), text_inputs[:,2,:].long()
        
        if self.use_finetune:
            text_outputs = self.roberta_model(input_ids, input_mask)
        else:
            with torch.no_grad():
                text_outputs = self.roberta_model(input_ids, input_mask)
        return text_outputs['last_hidden_state']
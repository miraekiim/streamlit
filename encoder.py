#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Cross Encoder fine tuning 시켰을때 성능 비교를 위해 추가

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


# In[2]:


class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.poly_m = kwargs['poly_m']
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
    
    def dot_attention(self, query, key, value):
        attention_weights = torch.matmul(query, key.transpose(2,1))
        attention_weights = F.softmax(attention_weights, -1)
        output = torch.matmul(attention_weights, value)
        return output
    
    def forward(self, context_input_ids, context_input_masks,
               responses_input_ids, responses_input_masks, labels = None):
        
        if labels is not None:
            responses_input_ids = responses_input_ids[:,0,:].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape
        
        
        # context encoder
        ctx_out = self.bert(context_input_ids, context_input_masks)[0]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out)
        
        # response encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        cand_emb = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1)
        
        # merge
        if labels is not None:
            cand_emb = cand_emb.permute(1,0,2)
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2])
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze()
            dot_product = (ctx_emb*cand_emb).sum(-1)
            mask = torch.eye(batch_size).to(context_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        
        else:
            ctx_emb = self.dot_attention(cand_emb, embs, embs)
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product


class CrossEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, text_input_ids, text_input_masks, text_input_segments, labels=None):
        batch_size, neg, dim = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(-1, dim)
        text_input_masks = text_input_masks.reshape(-1, dim)
        text_input_segments = text_input_segments.reshape(-1, dim)
        text_vec = self.bert(text_input_ids, text_input_masks, text_input_segments)[0][:,0,:]  # [bs,dim]
        score = self.linear(text_vec)
        score = score.view(-1, neg)
        if labels is not None:
            loss = -F.log_softmax(score, -1)[:,0].mean()
            return loss
        else:
            return score


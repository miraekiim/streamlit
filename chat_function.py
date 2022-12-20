#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from streamlit_chat import message
import pandas as pd
import pickle
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from encoder import PolyEncoder, CrossEncoder
from transform import SelectionJoinTransform, SelectionSequentialTransform, SelectionConcatTransform


# In[2]:
device = torch.device('cpu')
# pre-trained bert model
bert_name = 'bert-base-uncased'
bert_config = BertConfig.from_pretrained(bert_name)
# tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_name)
tokenizer.add_tokens(['\n'], special_tokens=True)

# 전역변수 설정
#global bert, model, rerank_model, context_transform, response_transform, concat_transform
# transfrom.py에서 input 함수 불러오기
context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=512)
response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=40)
concat_transform = SelectionConcatTransform(tokenizer=tokenizer, max_len=512)

# model 불러오기
# Poly Encoder model 경로
PATH = './model/poly_64_pytorch_model.bin'
# Cross Encoder model 경로
PATH1 = './model/cross_0_pytorch_model.bin'

# bert 모델 설정
bert = BertModel.from_pretrained(bert_name, config=bert_config)

# Poly Encoder model
model = PolyEncoder(bert_config, bert=bert, poly_m=64)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(PATH, map_location=device))
# Cross Encoder model
rerank_model = CrossEncoder(bert_config, bert=bert)
rerank_model.resize_token_embeddings(len(tokenizer))
rerank_model.load_state_dict(torch.load(PATH1, map_location=device))

def input_context(context):
    context_input_ids, context_input_masks = context_transform(context)
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]

    long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]

    contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)

    return contexts_token_ids_list_batch, contexts_input_masks_list_batch


def response_input(candidates):
    responses_token_ids_list, responses_input_masks_list = response_transform(candidates)
    responses_token_ids_list_batch, responses_input_masks_list_batch = [responses_token_ids_list], [responses_input_masks_list]

    long_tensors = [responses_token_ids_list_batch, responses_input_masks_list_batch]

    responses_token_ids_list_batch, responses_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)

    return responses_token_ids_list_batch, responses_input_masks_list_batch


def ctx_emb(contexts_token_ids_list_batch, contexts_input_masks_list_batch):
    with torch.no_grad():
        model.eval()
        
        ctx_out = model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(model.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, model.poly_m)
        poly_codes = model.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = model.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        return embs


def cands_emb(responses_token_ids_list_batch, responses_input_masks_list_batch):
    with torch.no_grad():
        model.eval()
                
        batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape # res_cnt is 1 during training
        responses_token_ids_list_batch = responses_token_ids_list_batch.view(-1, seq_length)
        responses_input_masks_list_batch = responses_input_masks_list_batch.view(-1, seq_length)
        cand_emb = model.bert(responses_token_ids_list_batch, responses_input_masks_list_batch)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        return cand_emb


def score(embs, cand_emb):
    with torch.no_grad():
        model.eval()

        ctx_emb = model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
        dot_product = (ctx_emb*cand_emb).sum(-1)
        
        return dot_product

# Cross Encoder 계산에 필요한 함수

def input_text(context,response):
    text_input_ids, text_input_masks, text_segment_ids = concat_transform(context,response)
    text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [text_input_ids], [text_input_masks], [text_segment_ids]
    
    long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]
    text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)
         
    return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch
    
def text_emb(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch):
    batch_size, neg, dim = text_token_ids_list_batch.shape
    text_token_ids_list_batch = text_token_ids_list_batch.reshape(-1,dim)
    text_input_masks_list_batch = text_input_masks_list_batch.reshape(-1,dim)
    text_segment_ids_list_batch = text_segment_ids_list_batch.reshape(-1,dim)
    text_vec = rerank_model.bert(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch)[0][:,0,:]
    
    model = rerank_model.linear
    score = model(text_vec)
    score = score.view(-1, 1)
    
    return score
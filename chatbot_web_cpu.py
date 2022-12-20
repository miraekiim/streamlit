#!/usr/bin/env python
# coding: utf-8

# In[11]:


# 필요한 라이브러리 및 모듈 설치
#!pip install streamlit
#!pip install streamlit_chat
#!pip install torch
#!pip install transformers


# In[3]:


# 필요한 모듈 및 라이브러리 불러오기
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
from chat_function import input_context, response_input, ctx_emb, cands_emb, score, input_text, text_emb

# 필요한 클래스 및 함수 정의하기
device = torch.device('cpu')
bert_name = 'bert-base-uncased'
bert_config = BertConfig.from_pretrained(bert_name)
tokenizer = BertTokenizer.from_pretrained(bert_name)
tokenizer.add_tokens(['\n'], special_tokens=True)
context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=512)
response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=40)
concat_transform = SelectionConcatTransform(tokenizer=tokenizer, max_len=512)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# caching을 해서 불러오면 한 번만 실행할 수 있음
# model과 데이터 불러올 때 사용
@st.cache(allow_output_mutation = True)
def cached_polymodel():
    
    device = torch.device('cpu')
    
    PATH = './model/poly_64_pytorch_model.bin'
    
    bert_name = 'bert-base-uncased'
    bert_config = BertConfig.from_pretrained(bert_name)
    bert = BertModel.from_pretrained(bert_name, config=bert_config)
    
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    tokenizer.add_tokens(['\n'], special_tokens=True)

    model = PolyEncoder(bert_config, bert=bert, poly_m=64)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(PATH, map_location=device))
    
    return model

@st.cache(allow_output_mutation = True)
def cached_crossmodel():
    
    device = torch.device('cpu')
    
    PATH = './model/cross_0_pytorch_model.bin'
    
    bert_name = 'bert-base-uncased'
    bert_config = BertConfig.from_pretrained(bert_name)
    bert = BertModel.from_pretrained(bert_name, config=bert_config)
    
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    tokenizer.add_tokens(['\n'], special_tokens=True)

    rerank_model = CrossEncoder(bert_config, bert=bert)
    rerank_model.resize_token_embeddings(len(tokenizer))
    rerank_model.load_state_dict(torch.load(PATH, map_location=device))
    
    return rerank_model

@st.cache(allow_output_mutation = True)
def get_dataset():
    
    # csv 파일 후보군 데이터
    data = pd.read_csv('./data/edcsdd_response.csv', encoding = 'utf8', index_col = 0)
    data = data.reset_index(drop=True)
    cand_data = data['response']
    
    # embedding 값 후보군 데이터
    with open('./data/edcsdd_cand_emb.pickle', 'rb') as fr:
        final_cand_emb = CPU_Unpickler(fr).load()       
        
    return cand_data, final_cand_emb

# 모델 및 데이터 로드
model = cached_polymodel()
rerank_model = cached_crossmodel()
cand_data, final_cand_emb = get_dataset()

st.header('BlueCare Consultant Bot')
st.markdown('Hello! I am a consultant bot')

# session으로 관리해야 초기화 되지 않음
# save chatbot utterances
if 'chatbot_reply' not in st.session_state:
    st.session_state['chatbot_reply'] = []

# save user's utterances
if 'user_utterance' not in st.session_state:
    st.session_state['user_utterance'] = []
    
with st.form('form', clear_on_submit = True):
    user_input = st.text_input('User: ', '')
    submitted = st.form_submit_button('send')

# chatbot과 대화하는 부분    
if submitted and user_input:
    idx_list = []
    top_cand = []
    
    user_context = [user_input]
    
    user_emb = ctx_emb(*input_context(user_context))
    final_score = score(user_emb, final_cand_emb)
    new_score = final_score.sort()
    
    # 후보군 5개 저장
    for i in range(5):
        idx_list.append(int(new_score[1][0][-5:][i]))
    for idx in idx_list:
        top_cand.append(cand_data[idx])
    
    top_cand = pd.Series(top_cand)
    
    # re-ranking
    rerank_list = []
    for i in range(len(top_cand)):
        response = [top_cand[i]]
        # score 계산
        cross_score = text_emb(*input_text(user_context, response))
        rerank_list.append(cross_score.item())
    
    final_index = rerank_list.index(max(rerank_list))
    best_response = top_cand[final_index]
    
    st.session_state.user_utterance.append(user_input)
    st.session_state.chatbot_reply.append(best_response)

# 대화 visualize
for i in range(len(st.session_state['user_utterance'])):
    message(st.session_state['user_utterance'][i], is_user = True, key = str(i) + '_user')
    if len(st.session_state['chatbot_reply']) > i:
        message(st.session_state['chatbot_reply'][i], key = str(i) + '_bot')
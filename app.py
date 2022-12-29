#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
import os
import os.path   
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertConfig, BertModel, BertTokenizer, AutoModel
from encoder import PolyEncoder
from transform import SelectionJoinTransform, SelectionSequentialTransform
import os
import sys
import urllib.request
import json
import pickle

#----------------------------------------------------------------------------#
# bin Download
#----------------------------------------------------------------------------#
#file = 'poly_16_pytorch_model_48.bin'     # 예제 Textfile

#if os.path.isfile(file):
#    print("Yes. it is a file")
#    pass
#else:
#    url = 'https://drive.google.com/uc?id=1e3eujIm3jjqCLL-nKk8FQmOtK0jARQ_U'
#    output = 'poly_16_pytorch_model_48.bin'
#    gdown.download(url, output, quiet=False)

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)

sp = pd.read_csv('static/token/token_filtering_plus_movie.csv')
sp_list = []
for i in range(len(sp['special token'])):
  sp_list.append(sp['special token'][i])
sp_list

# %%

#device = torch.("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# 모델이 github에 있을 경우 
#PATH = './poly_16_pytorch_model_32.bin'

#bert_name = 'klue/bert-base'
#bert_config = BertConfig.from_pretrained(bert_name)

#tokenizer = BertTokenizer.from_pretrained(bert_name)
#tokenizer.add_tokens(sp_list, special_tokens=True)

#context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=256)
#response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=128)

#bert = BertModel.from_pretrained(bert_name, config=bert_config)

#model = PolyEncoder(bert_config, bert=bert, poly_m=16)
#model.resize_token_embeddings(len(tokenizer))
#model.load_state_dict(torch.load(PATH))
#model.to(device)
#model.device

# %%
def context_input(context):
    context_input_ids, context_input_masks = context_transform(context)
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]

    long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]

    contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)

    return contexts_token_ids_list_batch, contexts_input_masks_list_batch


# %%
def response_input(candidates):
    responses_token_ids_list, responses_input_masks_list = response_transform(candidates)
    responses_token_ids_list_batch, responses_input_masks_list_batch = [responses_token_ids_list], [responses_input_masks_list]

    long_tensors = [responses_token_ids_list_batch, responses_input_masks_list_batch]

    responses_token_ids_list_batch, responses_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)

    return responses_token_ids_list_batch, responses_input_masks_list_batch


# %%
def embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch):

    with torch.no_grad():
        model.eval()
        
        ctx_out = model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(model.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, model.poly_m)
        poly_codes = model.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = model.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        return embs


# %%
def cand_emb_gen(responses_token_ids_list_batch, responses_input_masks_list_batch):

    with torch.no_grad():
        model.eval()
                
        batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape # res_cnt is 1 during training
        responses_token_ids_list_batch = responses_token_ids_list_batch.view(-1, seq_length)
        responses_input_masks_list_batch = responses_input_masks_list_batch.view(-1, seq_length)
        cand_emb = model.bert(responses_token_ids_list_batch, responses_input_masks_list_batch)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        return cand_emb

# %%
def loss(embs, cand_emb, contexts_token_ids_list_batch, responses_token_ids_list_batch):
    batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape

    ctx_emb = model.dot_attention(cand_emb, embs, embs) # [bs, bs, dim]
    # print(ctx_emb)
    ctx_emb = ctx_emb.squeeze()
    # print(ctx_emb)
    dot_product = (ctx_emb*cand_emb) # [bs, bs]
    # print(dot_product)
    dot_product = dot_product.sum(-1)
    print(dot_product)
    mask = torch.eye(batch_size).to(contexts_token_ids_list_batch.device) # [bs, bs]
    print(mask)
    loss = F.log_softmax(dot_product, dim=-1)
    print(loss)
    loss = loss * mask
    print(loss)
    loss = (-loss.sum(dim=1))
    print(loss)
    loss = loss.mean()
    print(loss)
    return loss

# %%
def score(embs, cand_emb):
    with torch.no_grad():
        model.eval()

        ctx_emb = model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
        dot_product = (ctx_emb*cand_emb).sum(-1)
        
        return dot_product


# %%
"""
### 데이터 검증
"""

# %% 피클생성
with open('static/pickle/train_data_source_hogwart_plus_mv_qin_sy_32.pickle', 'rb') as f:
    train = pickle.load(f)

# %%
"""
### 챗봇 테이블 생성
"""

# %% 피클생성시 주석 제거
data = {
    'context' : [],
    'response': []
}

# # %%
for sample in train:
    data['context'].append(sample['context'])
    data['response'].append([sample['responses'][0]])



# %% 피클 생성시 주석 제거

# # %%
df = pd.DataFrame(data)


# %% 피클 불러오기
with open('static/pickle/cand_embs.pickle', 'rb') as f:
    cand_embs = pickle.load(f)
#cand_embs.to(device)

# %%
"""
### generate context_embs
"""


 
# %%
"""
### Chatbot UI
"""
print('안녕하세요. 공감 만땅이~~⭐️ 공감이🍀 입니다.')

"""
파파고 api 활용해보기
"""
client_id = "1yruG5odcT_wqDo3WVnZ"
client_secret = "QjHIsoT4KJ"
########
#영어 함수
def entest(nums, best_answer):
    koText = urllib.parse.quote(nums)
    koTests = urllib.parse.quote(best_answer)

    # 여기서 en은 영어 ko는 한국어 if를 써서 둘이 바꾸면 될듯 아니면 리스트로 해서 때에 따라서 넣어주던가
    kodata = "source=ko&target=en&text=" + koText
    kodatas = "source=ko&target=en&text=" + koTests

    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)

    koresponse = urllib.request.urlopen(request, data=kodata.encode("utf-8"))
    koresponses = urllib.request.urlopen(request, data=kodatas.encode("utf-8"))

    korescode = koresponse.getcode()
    korescodes = koresponses.getcode()
    if(korescode==200):
        response_body = koresponse.read()
        response_bodys = koresponses.read()

        res = json.loads(response_body.decode('utf-8'))
        ress = json.loads(response_bodys.decode('utf-8'))
        ko_text = res['message']['result']['translatedText']
        ko_answer = ress['message']['result']['translatedText']
    return ko_answer, ko_text

#단일 한글 함수
def kotest(nums):
    koText = urllib.parse.quote(nums)

    # 여기서 en은 영어 ko는 한국어 if를 써서 둘이 바꾸면 될듯 아니면 리스트로 해서 때에 따라서 넣어주던가
    kodata = "source=en&target=ko&text=" + koText

    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)

    koresponse = urllib.request.urlopen(request, data=kodata.encode("utf-8"))

    korescode = koresponse.getcode()
    if(korescode==200):
        response_body = koresponse.read()

        res = json.loads(response_body.decode('utf-8'))
        ko_text = res['message']['result']['translatedText']
    return ko_text

#단일 영어 함수
def entext(best_answer):
    koTests = urllib.parse.quote(best_answer)

    # 여기서 en은 영어 ko는 한국어 if를 써서 둘이 바꾸면 될듯 아니면 리스트로 해서 때에 따라서 넣어주던가
    
    kodatas = "source=ko&target=en&text=" + koTests

    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)

    koresponses = urllib.request.urlopen(request, data=kodatas.encode("utf-8"))

    korescodes = koresponses.getcode()
    if(korescodes==200):
        response_bodys = koresponses.read()

        ress = json.loads(response_bodys.decode('utf-8'))
        ko_answer = ress['message']['result']['translatedText']
    return ko_answer
########

#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/', methods=['GET'])
def home():
    return render_template('forms/main.html')

@app.route("/helps")
def helps():
    return render_template('forms/help.html')

@app.route("/enhelps")
def enhelps():
    return render_template('forms/enhelps.html')

@app.route("/anitest")
def anitest(num=None):
    
    return render_template('forms/ani.html')

@app.route('/calculate', methods=['POST', 'GET'])
def calculate(num=None):
    if request.method == 'POST':

        query = [request.form['num']]
        nums = request.form['num']
        
        en = request.form['fruit']
        print(query)
        
        best_num = -1
        embs = embs_gen(*context_input(query))
        s = score(embs, cand_embs)
        idx = int(s[0].sort()[-1][best_num])
        best_answer = df['response'][idx][0]

        print('공감이🍀 : ')
        print(best_answer)

        if en == '영어':
            ko_answer, ko_text = entest(nums, best_answer)

            if '디멘터' in nums:
                print(1)
                return render_template('forms/enani.html', num=ko_answer, chart=ko_text)
            elif '늑대인간' in nums:
                print(1.5)
                return render_template('forms/enani.html', num=ko_answer, chart=ko_text)
            elif '보름달' in nums:
                print(0.5)
                return render_template('forms/enani.html', num=ko_answer, chart=ko_text)
            else:
                print(2)
                return render_template('forms/enindex.html', num=ko_answer, chart=ko_text)
        else:
            if '디멘터' in nums:
                print(1)
                return render_template('forms/ani.html', num=best_answer, chart=nums)
            elif '늑대인간' in nums:
                print(1.5)
                return render_template('forms/ani.html', num=best_answer, chart=nums)
            elif '보름달' in nums:
                print(0.5)
                return render_template('forms/ani.html', num=best_answer, chart=nums)
            else:
                print(2)
                return render_template('forms/index.html', num=best_answer, chart=nums)
    if num==None:
        print(0)
        return render_template('forms/index.html', num=num)

# # 영어에서 한글로
@app.route("/thort", methods=['POST', 'GET'])
def thort(num=None):
    if request.method == 'POST':

        query = [request.form['num']]
        nums = request.form['num']
        
        en = request.form['fruit']
        print(query)

        if en == 'Korean':
            ko_text = [kotest(nums)]
            ko_tests = kotest(nums)

            best_num = -1
            embs = embs_gen(*context_input(ko_text))
            s = score(embs, cand_embs)
            idx = int(s[0].sort()[-1][best_num])
            best_answer = df['response'][idx][0]

            print('공감이🍀 : ')
            print(best_answer)

            if 'dement' in nums:
                print(1)
                return render_template('forms/ain.html', num=best_answer, chart=ko_tests)
            elif 'werewolf' in nums:
                print(1.5)
                return render_template('forms/ani.html', num=best_answer, chart=ko_tests)
            elif 'full moon' in nums:
                print(0.5)
                return render_template('forms/ani.html', num=best_answer, chart=ko_tests)
            else:
                print(2)
                return render_template('forms/index.html', num=best_answer, chart=ko_tests)
        else:
            ko_text = [kotest(nums)]

            best_num = -1
            embs = embs_gen(*context_input(ko_text))
            s = score(embs, cand_embs)
            idx = int(s[0].sort()[-1][best_num])
            best_answer = df['response'][idx][0]

            print('공감이🍀 : ')
            print(best_answer)

            en_best_answer = entext(best_answer)

            if 'Dement' in nums:
                print(1)
                return render_template('forms/enani.html', num=en_best_answer, chart=nums)
            elif 'Dementor' in nums:
                print(1.7)
                return render_template('forms/enani.html', num=en_best_answer, chart=nums)
            elif 'werewolf' in nums:
                print(1.5)
                return render_template('forms/enani.html', num=en_best_answer, chart=nums)
            elif 'full moon' in nums:
                print(0.5)
                return render_template('forms/enani.html', num=en_best_answer, chart=nums)
            else:
                print(2)
                return render_template('forms/enindex.html', num=en_best_answer, chart=nums)
    if num==None:
        print(0)
        return render_template('forms/enindex.html', num=num)

# https://url.com/coffe
@app.route('/hog', methods=['POST'])
def coffe():
    req = request.get_json()
    print(req)
    # 카카오에 저장된 엔티티 위치?
    # coffe_menu = req["action"]["detailParams"]["coffe_menu"]["value"]
    coffe_menu = req["userRequest"]["utterance"]
    print(coffe_menu)
    query = [coffe_menu]
    print(query)
    best_num = -1
    embs = embs_gen(*context_input(query))
    s = score(embs, cand_embs)
    idx = int(s[0].sort()[-1][best_num])
    best_answer = df['response'][idx][0]

    print('공감이🍀 : ')
    print(best_answer)
    answer = best_answer
    # answer = "아메리카노"

    # 답변 설정
    res = {
        "version" : "2.0",
        "template" : {
            "outputs" : [
                {
                    "simpleText" : {
                        "text" : answer
                    }
                }
            ]
        }
    }

    # 답변 발사!
    return jsonify(res)

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''

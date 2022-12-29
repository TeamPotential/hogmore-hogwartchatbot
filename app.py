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



#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/', methods=['GET'])
def home():
    return render_template('forms/main.html')

@app.route('/calculate')
def calculate(num=None):
    return render_template('forms/index.html', num=num)

@app.route("/helps")
def helps():
    return render_template('forms/help.html')

@app.route("/enhelps")
def enhelps():
    return render_template('forms/enhelps.html')

@app.route("/anitest")
def anitest(num=None):
    
    return render_template('ani.html')

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

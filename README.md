# Hogmore : Hogwart chatbot based on Poly-Encoder
 
## 진행기간 
- 2022.11.14 ~ 2022.12.21

## 개요

- 해리포터 세계관에 몰입하고 싶은 사람들을 위한 챗봇을 만들기 위해서 Poly-Encoder 기반 해리포터의 세계관을 담은 Closed Domain Chatbot 제작하여 Flask를 사용해 Web 페이지와 Kakao API에 적용
       
  
## 팀 구성 및 역할   

- 한민재 : 모델 구축, 데이터 생성
- 정가영 : 데이터 분석, 데이터 생성
- 이성연 : 모델 핸들링, 데이터 생성
- 양병진 : Flask, 데이터 생성
- 김규인 : 데이터 전처리, 데이터 생성
___
## 폴더 설명

- data - 영화 대사
  - 모델 학습에 사용된 데이터셋
  
- data_preprocessing
  - 데이터 전처리에 사용된 코드
  
- train_model
  - 모델 학습에 사용된 코드    



## 사용된 데이터 

- 해리포터 영화 전 시리즈 대사
- 포터모어 기반 문답 데이터

## 과정  

 1. 개발환경 : Python, PyTorch, Colab, AWS(Amazon Web Service), html, Flask, Kakao API
 
 2. 데이터 전처리
   - 데이터 생성
     
      해리포터 세계관 사전인 포터모어를 기반으로 문답 데이터 생성
         
   - 스페셜 토큰 : 454종류
   
       호그와트, 호그스미드 같이 고유명사 이면서 형태가 비슷한 단어에 대해 구별하지 못하는 문제 발생하여  소설 속 고유명사들을  토큰으로 지정해주는 작업 수행 
          
   - 증강  
     [Pororo](https://github.com/kakaobrain/pororo) 라이브러리의 유의어 변환 기능을 사용하여 수기 데이터를 증강 후 의미가 심하게 훼손된 문장은 변경 및 삭제 처리
           
___


   3. 데이터셋 조합
   
 데이터셋 | 데이터 갯수 | 
 :-------:|:-----------:|
 정제데이터(수기+유의어) | 5,945 |         
 오타데이터 + 정제데이터 | 20,878 |       
 수기 + 유의어증강 데이터 | 5,950 |         
 수기 + 유의어증강+세계관 | 6,538 |        
 
 
 
 4. 데이터셋 분석
  


 ![image](https://user-images.githubusercontent.com/112064534/209896472-33cbc59c-2baa-497a-ac07-c7bcea0aecbc.png)

데이터 복잡도를 낮추기 위해서 문장 속 같은 의미, 다른 형태의 고유 명사들을 대표적인 한 단어로 통일하여 정제하거나 복잡도를 높이기 위해서 오타를 늘리게 되면 수기데이터와 유의어를 증강한 데이터로 학습하여 출력되는 것보다 정확도가 떨어짐. 
따라서 데이터 고유의 다양성을 주기 위해서 해리포터 세계관을 좀 더 디테일하게 체험할 수 있도록 기존 영화 대사 데이터에서 세계관을 나타내는 데이터를 기반으로 질적으로 다양하게 데이터를 생성하여 수기데이터와 유의어 증강 데이터와 합쳐서 학습을 했을 때 성능이 가장 좋았음. 




 ## 모델

![스크린샷_20221229_112250](https://user-images.githubusercontent.com/113493695/209895755-b6d692a7-170d-4bd7-9b4f-81d6ffcf5e97.png)

 - Transformer : Model trained from scratch
 - GPT2 : ‘skt/kogpt2-base-v2’
 - Poly-Encoder : ‘klue/bert-base’
 
 
  데이터 셋 구조는 다음과 같이 하나의 질문에 하나의 정답과 N개의 오답이 라벨링 되는 형태로 이루어짐.
![스크린샷_20221229_112431](https://user-images.githubusercontent.com/113493695/209895780-d7e4ea60-abd7-42b9-931a-77bd09fd582c.png)

 
 
![스크린샷_20221229_112742](https://user-images.githubusercontent.com/113493695/209895818-16cd0ce0-e414-4952-bb3f-442eb9d39185.png)

 출력된 yctxt,cand 를 스칼라 값으로 만들어 주기 위한 W 를 곱하여 스코어 계산하여 유사도 점수 뽑아냄.
 cross-entropy는 모델에서 예측한 확률과 정답 확률을 모두 사용해 측정한 값으로, cross-entropy를 최소화하기 위해 negative 방식을 사용.


 유사도를 사용하는 모델에는 Bi-encoder와 Cross-encoder가 있음.
![스크린샷_20221229_113859](https://user-images.githubusercontent.com/113493695/209896367-eb94cd04-f3a5-4cd3-98b2-9e45f4317fe2.png)

 Bi-encoder는 Candidate Embedding을 미리 산출해 저장 가능하여서 다시 Embedding을 산출하는 과정이 필요 없어지고 Retrieval system에 적용하여 계산속도가 빨라지는 효과를 얻을 수 있음.
 
![스크린샷_20221229_113914](https://user-images.githubusercontent.com/113493695/209896555-f267ed31-6fa3-400e-998a-0431beb40e6f.png)

Cross-encoder는 유사도 점수를 구할 때 Candidate Embedding을 매번 다시 산출하여 번거롭지만 보다 연관성 있는 Embedding이 가능하지만 계산 속도가 느려짐.

![스크린샷_20221229_113933](https://user-images.githubusercontent.com/113493695/209896687-8b31e7e7-ceeb-42be-82cb-8cd3f3d3c7ae.png)

 - Candidate Embedding을 미리 산출하여 Retrieval system에 적용 가능.
 - Attention 시 Context와 Candidate가 함께 수행하여 유사도 점수 계산 능력이 높아짐.
 - Poly-M 값에 따라 Attention의 갯수가 늘어나게 되는데 값이 높아질수록 유사도 점수 계산의 정확도가 높아지지만 연산량이 많아짐.

## 평가지표

- Rank-less recommendation metrics : 오답과 함께 우선순위까지 고려하는 지표

- R@K (Recall at K) : K개를 추천했을 때, 추천 되어야 했을 관련있는 정답과 겹치는 비율을 나타내는 지표

- RR (Reciprocal Rank) : 추천 리스트 중, 사용자가 실제로 선호하는 가장 높은 순위의 역수

- MRR (Mean Reciprocal Rank) : 하나의 추천 리스트에서, 각 사용자들이 갖는 reciprocal rank의 평균을 구한 지표

- 질문 리스트 : 직접 질문 리스트 만들어 모델에 Inference 모델의 성능을 직접 확인

## 결과
- Closed Domain Chatbot을 통한 호그와트 세계관을 구현 시도
- 답변의 정확도와 속도가 가장 좋다고 판단되는 Poly-Encoder 모델을 확정
- 완성된 호그모어 모델을 Web 페이지 및 Kakao API를 통해 어플화
  

## 보완점
- 더 넓은 범위의 대화가 가능하도록 다양한 주제의 문답 데이터 추가
- 더욱 정교한 대화가 가능한 장기기억 챗봇으로 개발  

## 참조
-Humeau, S., Shuster, K., Lachaux, M. A., & Weston, J. (2019). Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring. arXiv preprint arXiv:1905.01969.
https://doi.org/10.48550/arXiv.1905.01969

https://github.com/chijames/Poly-Encoder






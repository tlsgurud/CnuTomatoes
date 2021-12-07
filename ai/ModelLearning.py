# Generate A.I Model
# >> Model: 긍부정 분석 모델(감성분석)
# >> Module: Tensorflow, Keras
# >> Dataset: Naver Sentiment Movie Corpus(https://github.com/e9t/nsmc/)


#################
# Dataset Intro #
#################

# 데이터셋: Naver Sentiment Movie Corpus(https://github.com/e9t/nsmc/)
# >> 네이버 영화 리뷰 중 영화당 100개의 리뷰를 모아
# >> 총 200,000개의 리뷰(훈련: 15만개, 테스트 5만개)로
# >> 이루어져잇고, 1~10점까지의 평점 중 중립적인 평점(5~8)은
# >> 제외하고  1~4점을 부정, 9~10점을 긍정으로 동일한 비율로
# >> 데이터에 포함시킴

# >> 데이터는 id, document, label 세개의 열로 이루어져있음
# >> id: 리뷰의 고유한 Key값
# >> document: 리뷰의 내용
# >> label: 긍정(1)인지 부정(0)인지 나타냄
#           평점이 긍정 (9~10점), 부정(1~4점), 5~8점은 제거


import json
import os
import nltk
# import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import font_manager, rc
from pprint import pprint
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


#############
# File Open #
#############

 # ~.txt 파일에서 데이터를 불러오는 method
def read_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:  # with => 자동으로 close
        data = [line.split('\t') for line in f.read().splitlines()]  # line = ["4059827",  "배우들이 아깝다",    "0"]
        data = data[1:]  # 제목열 제외
    return data


    # data = []
        # for line in f.read().splitlines():
        #     data.append(line.split('\t'))


# nsmc 데이터를 불러와서 python 변수에 담기
train_data = read_data('./dataset/ratings_train.txt')  # 트레이닝 데이터 Open
test_data = read_data('./dataset/ratings_test.txt')    # 테스트 데이터 Open

# print(len(train_data))
# print(train_data[0])
#
# print(len(test_data))
# print(test_data[0])


##############
# Preprocess #
##############
# 데이터를 학습하기에 알맞게 처리해보자. konlpy 라이브러리를 사용해서
# 형태소 분석 및 품사 태강을 진행한다. 네이버 영화 데이터는
# 맞춤법이나 띄어쓰기가 제대로 되어있지 않은 경우가 있기 때문에
# 정확한 분류를 위해서 konlpy를 사용한다.
# konlpy에는 여러 클래스가 존재하지만 그 중 okt(open korean text)
# 사용하여 간단한 문장분석을 실행한다.
okt = Okt()

# print(okt.pos('이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요요'))


# Train, Test 데이터셋에 형태소 분석과 품사 태깅 작업 진행
# norm : 그래욬ㅋㅋ -> 그래요
# stem : 원형을 찾음 ( 그래요 -> 그렇다)
def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


# train.docs.json 저장
# test.docs.json 저장

if os.path.isfile('train_docs.json'):
    # 전처리 작업이 완료된 train_docs.json 파일이 있을 때
    # train_docs.json과 test_docs.json 파일 로드!
    with open('train_docs.json', 'r', encoding='UTF-8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', 'r', encoding='UTF-8') as f:
        test_docs = json.load(f)
else:
    # 전처리 된 파일이 없을 때
    # 전처리 작업 시작!
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # 전처리 완료 => JSON 파일로 저장
    with open('train_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent='\t')
    with open('test_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent='\t')

# 전처리 작업 데이터 확인
# pprint(train_docs[0])
# pprint(test_docs[0])
# print(len(train_docs))
# print(len(test_docs))

# 분석한 데이터의 토큰(문자열 분석을 휘한 작은 단위)의 갯수를 확인
tokens = [t for d in train_docs for t in d[0]]
# print(len(tokens))

# 이 데이터를 nltk 라이브러리를 통해서 전처리,
# vocab().most_common을 이용해서 가장 자주 사용되는 단어 빈도수 확인
text = nltk.Text(tokens, name='NSMC')

# 전체 토큰의 개수
# print(len(text.tokens))

# 중복을 제외한 토큰의 개수
# print(len(set(text.tokens)))      # set() => 중복제거

# 출현 빈도가 높은 상위 토근 10개
# pprint(text.vocab().most_common(10))

# 자주 출현하는 단어 50개를 matplotlib을 통해 그래프로 그리기
# 한글폰트 설정을 해줘야 깨지지 않고 출력됨
# font_name = '/Users/shkabcd/Library/Fonts/Arial Unicode.ttf'
#
# font_name = font_manager.FontProperties(fname=font_name).get_name()
# rc('font', family = font_name)
#
# plt.figure(figsize=(20,10))
# text.plot(50)
# plt.show


# 자주 사옹되는 토큰 5000개를 사용해서 데이터를 벡터화 시킨다
# 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어
# BOW(Bag of Words) 인코딩한 벡터를 만드는 역할을 한다.
select_words = [f[0] for f in text.vocab().most_common(5000)]
print('type:', type(select_words))
print('len:', len(select_words))
print('data:', select_words[:10])

if os.path.isfile('selectword.txt') == False:
    f = open('selectword.txt', 'w', encoding='UTF-8')
    print('>> LOG: selectword 파일 저장 시작')
    for i in select_words:
        i += '\n'  # \n == 한 줄 개행(enter)
        f.write(i)
    f.close()
    print('>> LOG: 파일 저장 완료')


#################
# Deep Learning #
#################

# Train / Test 데이터 정의
def term_frequency(doc):
    return [doc.count(word) for word in select_words]


# > Train(학습) 데이터 정의
train_x = [term_frequency(d) for d, _ in train_docs]  # _ => 안쓰겠다.
train_y = [c for _, c in train_docs]
# > Test(검증) 데이터 정의
test_x = [term_frequency(d) for d, _ in test_docs]
test_y = [c for _, c in test_docs]

# 이제 데이터를 float로 형 변환 시켜주면 데이터 전처리 과정을 끝
print('>> 형변환 시작')
x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

# AI 모델 정의
model = models.Sequential()  # 모델 생성

# > 모델 층(layer)을 구성
model.add(layers.Dense(64, activation='relu', input_shape=(5000,)))  # 1층 생성
model.add(layers.Dense(64, activation='relu'))  # 2층 생성
model.add(layers.Dense(1, activation='sigmoid'))  # 3층 생성


# AI 모델 훈련 설정
# 1) Optimaizer: 훈련과정을 설정
# 2) loss: 최적화 과정에서 최소화 될 손실 함수를 설정
# 3) metrics: 훈련을 모니터링하기 위해 사용
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# AI 모델 훈련(학습) : 15만
model.fit(x_train, y_train, epochs=10, batch_size=512)
# AI 모델 훈련(검증) : 5만
results = model.evaluate(x_test, y_test)
print(results)
# AI 모델 저장
model.save('my_model.h5')
print('Trained Model Saved.')

# / => 하위폴더
# .. => 상위폴더
# . => 현재폴더

# ./dataset/ratings_train.txt


# 절대경로와 상대경로
# C:/cnu_workspace/CnuTomatoes
#                      ㄴ ai
#                          ㄴ dataset
#                                ㄴ ratings_text.txt
#                                ㄴ ratings_train.txt
#                          ㄴ ModelLearning.py
#                      ㄴ model
#                      ㄴ webcrawl
#                      ㄴ main.py
#                      ㄴ README.md






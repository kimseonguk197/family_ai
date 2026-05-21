import os
import re
import urllib.request
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

import pandas as pd
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from tqdm import tqdm


# ================================
# 0. 초소형 영어 Word2Vec 실습
# ================================

from gensim.models import Word2Vec

print("=== 초소형 Word2Vec 실습 시작 ===")

# 간단한 예제 문장 10개
small_sentences = [
    ["man", "is", "person"],
    ["woman", "is", "person"],
    ["boy", "is", "young", "man"],
    ["girl", "is", "young", "woman"],
    ["guy", "is", "man"],
    ["lady", "is", "woman"],
    ["gentleman", "is","man"],
    ["soldier", "is", "man"],
    ["poet", "is", "man"],
    ["kid", "is", "young", "person"]
]

print(f"학습 문장 개수: {len(small_sentences)}")
print("학습 문장:")
for i, sentence in enumerate(small_sentences, 1):
    print(f"{i}. {sentence}")

# 소규모 Word2Vec 학습
small_model = Word2Vec(
    sentences=small_sentences,
    vector_size=20,
    window=2,
    min_count=1,
    workers=1,
    sg=0,
    seed=42
)

print("\n=== 소규모 학습 완료 ===")

# 단어 목록 확인
print("학습된 단어들:")
print(list(small_model.wv.index_to_key))

# 'man'과 유사한 단어 확인
print("\n[man과 유사한 단어]")
print(small_model.wv.most_similar("man"))

# 'woman'과 유사한 단어 확인
print("\n[woman과 유사한 단어]")
print(small_model.wv.most_similar("woman"))


print("=== 대규모 영어 데이터 준비 시작 ===")
xml_file = "ted_en.xml"

# 파일이 없을 때만 다운로드
if not os.path.exists(xml_file):
    print("ted_en.xml 파일이 없어서 다운로드를 시작합니다.")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml",
        filename=xml_file
    )
    print("데이터 다운로드 완료")
else:
    print("ted_en.xml 파일이 이미 존재하므로 다운로드를 건너뜁니다.")

# XML 파일 파싱
with open(xml_file, 'r', encoding='UTF8') as targetXML:
    target_text = etree.parse(targetXML)

# <content> 내용만 추출
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 괄호 안 내용 제거 (예: (Laughter))
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 문장 토큰화
sent_text = sent_tokenize(content_text)

print(f"문장 개수: {len(sent_text)}")

# 정규화 (소문자 + 특수문자 제거)
normalized_text = []
for sentence in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", sentence.lower())
    normalized_text.append(tokens)

# 단어 토큰화
result = [word_tokenize(sentence) for sentence in normalized_text]

print(f"총 샘플 개수: {len(result)}")
print("샘플 3개:")
for line in result[:3]:
    print(line)

# Word2Vec 학습
print("\n=== Word2Vec 학습 시작 ===")
model_en = Word2Vec(
    sentences=result,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=0
)

print("학습 완료")

# 유사 단어 확인
print("\n[man과 유사한 단어]")
print(model_en.wv.most_similar("man"))

# 모델 저장
model_en.wv.save_word2vec_format('eng_w2v')

# 모델 로드
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")

print("\n[로드된 모델로 다시 확인]")
print(loaded_model.most_similar("man"))

# ===========================================
# # 2. 한국어 Word2Vec 실습 : 오래걸리므로 생략해도 될듯
# ===========================================
# print("\n=== 한국어 데이터 다운로드 시작 ===")

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt",
#     filename="ratings.txt"
# )

# # 데이터 로드
# train_data = pd.read_table('ratings.txt')

# print("전체 데이터 개수:", len(train_data))

# # 결측값 제거
# train_data = train_data.dropna(how='any')
# print("결측값 제거 후:", len(train_data))

# # 한글 외 제거
# train_data['document'] = train_data['document'].str.replace(
#     "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True
# )

# # 불용어 정의
# stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# # 형태소 분석기
# okt = Okt()

# print("\n=== 토큰화 시작 (시간 소요됨) ===")

# tokenized_data = []
# for sentence in tqdm(train_data['document']):
#     tokens = okt.morphs(sentence, stem=True)
#     tokens = [word for word in tokens if word not in stopwords]
#     tokenized_data.append(tokens)

# print("토큰화 완료")

# # 리뷰 길이 분석
# lengths = [len(review) for review in tokenized_data]

# print("최대 길이:", max(lengths))
# print("평균 길이:", sum(lengths) / len(lengths))

# # 히스토그램 출력
# plt.hist(lengths, bins=50)
# plt.xlabel('length')
# plt.ylabel('count')
# plt.title('Review Length Distribution')
# plt.show()

# # Word2Vec 학습
# print("\n=== 한국어 Word2Vec 학습 ===")

# model_kr = Word2Vec(
#     sentences=tokenized_data,
#     vector_size=100,
#     window=5,
#     min_count=5,
#     workers=4,
#     sg=0
# )

# print("학습 완료")

# # 임베딩 크기 확인
# print("임베딩 매트릭스 shape:", model_kr.wv.vectors.shape)

# # 유사 단어 출력
# print("\n[최민식과 유사한 단어]")
# print(model_kr.wv.most_similar("최민식"))

# print("\n[히어로와 유사한 단어]")
# print(model_kr.wv.most_similar("히어로"))
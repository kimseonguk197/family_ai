import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()


# 총 문서의 수
N = len(docs) 

def tf(t, d):
  return d.count(t)

def idf(t):
  df = 0
  for doc in docs:
    df += t in doc
  return log(N/(df+1))

def tfidf(t, d):
  return tf(t,d)* idf(t)

result = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)

print("--- TF (Term Frequency) Matrix ---")
print(tf_)


result_tfidf = []
for i in range(N):
    result_tfidf.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        # 미리 정의하신 tfidf 함수를 호출하여 값을 계산합니다.
        result_tfidf[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result_tfidf, columns = vocab)

print("\n--- TF-IDF Matrix ---")
print(tfidf_)
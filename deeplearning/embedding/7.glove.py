import re
import math
import random
import urllib.request
import os
from collections import Counter, defaultdict

import numpy as np
from lxml import etree
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')


# ======================================
# 1. 초소형 예제로 동시등장 확률 확인
# ======================================

toy_sentences = [
    ["i", "like", "deep", "learning"],
    ["i", "like", "nlp"],
    ["i", "enjoy", "flying"]
]

def build_vocab(sentences):
    vocab = sorted(set(word for sent in sentences for word in sent))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_id

def build_co_matrix(sentences, word_to_id, window_size=1):
    vocab_size = len(word_to_id)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    for sent in sentences:
        for i, center_word in enumerate(sent):
            center_id = word_to_id[center_word]
            left = max(0, i - window_size)
            right = min(len(sent), i + window_size + 1)

            for j in range(left, right):
                if i == j:
                    continue
                context_word = sent[j]
                context_id = word_to_id[context_word]
                matrix[center_id, context_id] += 1
    return matrix

def p_k_given_i(matrix, word_to_id, i_word, k_word):
    i = word_to_id[i_word]
    k = word_to_id[k_word]
    row_sum = matrix[i].sum()
    return 0 if row_sum == 0 else matrix[i, k] / row_sum
def top_context_words(matrix, word_to_id, id_to_word, query, topn=5):
    if query not in word_to_id:
        return []

    row = matrix[word_to_id[query]]
    result = []

    for idx, count in enumerate(row):
        if count > 0:
            result.append((id_to_word[idx], float(count)))

    result.sort(key=lambda x: x[1], reverse=True)
    return result[:topn]

vocab, word_to_id = build_vocab(toy_sentences)
co_matrix = build_co_matrix(toy_sentences, word_to_id, window_size=1)

print("=== 동시등장 행렬 ===")
print("vocab:", vocab)
print(co_matrix)

print("\n=== 조건부 확률 비율 확인 ===")
for k in ["learning", "enjoy", "like", "nlp"]:
    p1 = p_k_given_i(co_matrix, word_to_id, "deep", k)
    p2 = p_k_given_i(co_matrix, word_to_id, "flying", k)
    ratio = (p1 + 1e-12) / (p2 + 1e-12)
    print(f"k={k:10s} | P({k}|deep)={p1:.4f} | P({k}|flying)={p2:.4f} | ratio={ratio:.4e}")


id_to_word = {i: w for w, i in word_to_id.items()}
print("learning의 주변 단어:", top_context_words(co_matrix, word_to_id, id_to_word, "learning"))
print("deep의 주변 단어:", top_context_words(co_matrix, word_to_id, id_to_word, "deep"))
print("like의 주변 단어:", top_context_words(co_matrix, word_to_id, id_to_word, "like"))

# # ======================================
# # 2. 최소 GloVe 구현 : 아래는 복잡하니 생략
# # ======================================

# class SimpleGloVe:
#     def __init__(self, vector_size=50, learning_rate=0.05, x_max=100, alpha=0.75, seed=42):
#         self.vector_size = vector_size
#         self.learning_rate = learning_rate
#         self.x_max = x_max
#         self.alpha = alpha
#         self.seed = seed

#     def build_vocab(self, sentences, min_count=1):
#         counter = Counter()
#         for sent in sentences:
#             counter.update(sent)

#         vocab = [w for w, c in counter.items() if c >= min_count]
#         self.word_to_id = {w: i for i, w in enumerate(vocab)}
#         self.id_to_word = {i: w for w, i in self.word_to_id.items()}
#         self.vocab_size = len(vocab)

#     def build_cooccurrence(self, sentences, window_size=5):
#         cooccurrence = defaultdict(float)

#         for sent in sentences:
#             sent = [w for w in sent if w in self.word_to_id]
#             for i, center_word in enumerate(sent):
#                 center_id = self.word_to_id[center_word]

#                 left = max(0, i - window_size)
#                 right = min(len(sent), i + window_size + 1)

#                 for j in range(left, right):
#                     if i == j:
#                         continue
#                     context_word = sent[j]
#                     context_id = self.word_to_id[context_word]

#                     distance = abs(i - j)
#                     cooccurrence[(center_id, context_id)] += 1.0 / distance

#         return cooccurrence

#     def weight_func(self, x):
#         return (x / self.x_max) ** self.alpha if x < self.x_max else 1.0

#     def fit(self, sentences, min_count=5, window_size=5, epochs=10):
#         random.seed(self.seed)
#         np.random.seed(self.seed)

#         self.build_vocab(sentences, min_count=min_count)
#         cooccurrence = self.build_cooccurrence(sentences, window_size=window_size)
#         co_items = list(cooccurrence.items())

#         self.W = (np.random.rand(self.vocab_size, self.vector_size) - 0.5) / self.vector_size
#         self.C = (np.random.rand(self.vocab_size, self.vector_size) - 0.5) / self.vector_size
#         self.bw = np.zeros(self.vocab_size)
#         self.bc = np.zeros(self.vocab_size)

#         for epoch in range(epochs):
#             random.shuffle(co_items)
#             total_loss = 0

#             for (i, j), x_ij in co_items:
#                 weight = self.weight_func(x_ij)
#                 log_x = math.log(x_ij)

#                 pred = np.dot(self.W[i], self.C[j]) + self.bw[i] + self.bc[j]
#                 diff = pred - log_x
#                 loss = weight * (diff ** 2)
#                 total_loss += loss

#                 grad = weight * diff
#                 w_i = self.W[i].copy()
#                 c_j = self.C[j].copy()

#                 self.W[i] -= self.learning_rate * grad * c_j
#                 self.C[j] -= self.learning_rate * grad * w_i
#                 self.bw[i] -= self.learning_rate * grad
#                 self.bc[j] -= self.learning_rate * grad

#             print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss:.4f}")

#     def get_vector(self, word):
#         idx = self.word_to_id[word]
#         return self.W[idx] + self.C[idx]

#     def most_similar(self, word, topn=10):
#         target = self.get_vector(word)
#         target_norm = np.linalg.norm(target)

#         result = []
#         for other_word in self.word_to_id:
#             if other_word == word:
#                 continue
#             vec = self.get_vector(other_word)
#             sim = np.dot(target, vec) / (target_norm * np.linalg.norm(vec) + 1e-12)
#             result.append((other_word, float(sim)))

#         result.sort(key=lambda x: x[1], reverse=True)
#         return result[:topn]


# # ======================================
# # 3. TED 데이터 준비
# # ======================================

# xml_file = "ted_en.xml"

# if not os.path.exists(xml_file):
#     urllib.request.urlretrieve(
#         "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml",
#         filename=xml_file
#     )

# with open(xml_file, "r", encoding="UTF8") as f:
#     xml = etree.parse(f)

# text = "\n".join(xml.xpath("//content/text()"))
# text = re.sub(r"\([^)]*\)", "", text)

# sentences = sent_tokenize(text)
# sentences = [re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() for s in sentences]
# sentences = [word_tokenize(s) for s in sentences if s]
# sentences = [s for s in sentences if len(s) > 1]

# print(f"\n문장 수: {len(sentences)}")


# # ======================================
# # 4. GloVe 학습 및 확인
# # ======================================

# model = SimpleGloVe(vector_size=100, learning_rate=0.05, x_max=100, alpha=0.75, seed=42)
# model.fit(sentences, min_count=5, window_size=5, epochs=10)

# print("\n[man과 유사한 단어]")
# print(model.most_similar("man"))

# print("\n[woman과 유사한 단어]")
# print(model.most_similar("woman"))
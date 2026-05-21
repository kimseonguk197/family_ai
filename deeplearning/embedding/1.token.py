# ===============================
# 1. 라이브러리 import
# ===============================
from nltk.tokenize import word_tokenize, WordPunctTokenizer, sent_tokenize
from nltk.tag import pos_tag
import nltk
nltk.download('punkt')   # 토큰화
nltk.download('averaged_perceptron_tagger_eng')  # 품사 태깅
nltk.download('wordnet')  # 정규화/lemmatization
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from konlpy.tag import Okt, Kkma

# ===============================
# 2. 영어 단어 토큰화 비교
# ===============================
text_en = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."

print("=== 영어 단어 토큰화 비교 ===")

# (1) NLTK word_tokenize
print("\n[1] word_tokenize 결과:")
print(word_tokenize(text_en))

# (2) WordPunctTokenizer
print("\n[2] WordPunctTokenizer 결과:")
print(WordPunctTokenizer().tokenize(text_en))

# (3) Keras text_to_word_sequence
print("\n[3] text_to_word_sequence 결과:")
print(text_to_word_sequence(text_en))


# ===============================
# 3. Penn Treebank Tokenizer
# ===============================
text_treebank = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."

print("\n=== Penn Treebank 토큰화 ===")
tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(text_treebank))


# ===============================
# 4. 문장 토큰화
# ===============================
text_sentence = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy."

print("\n=== 문장 토큰화 ===")
print(sent_tokenize(text_sentence))


# ===============================
# 5. 품사 태깅 (POS Tagging)
# ===============================
print("\n=== 영어 품사 태깅 ===")

tokens = word_tokenize(text_en)
print("토큰:", tokens)

pos = pos_tag(tokens)
print("품사 태깅:", pos)


# ===============================
# 6. 한국어 형태소 분석
# ===============================
text_ko = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"

print("\n=== 한국어 형태소 분석 (Okt) ===")
okt = Okt()
print("형태소:", okt.morphs(text_ko))
print("품사:", okt.pos(text_ko))
print("명사:", okt.nouns(text_ko))


print("\n=== 한국어 형태소 분석 (Kkma) ===")
kkma = Kkma()
print("형태소:", kkma.morphs(text_ko))
print("품사:", kkma.pos(text_ko))
print("명사:", kkma.nouns(text_ko))
# ===============================
# 정제(Cleaning) & 정규화(Normalization) 통합 예제
# ===============================

import re
from collections import Counter

print("=" * 60)
print("1. 원본 텍스트")
print("=" * 60)

text = """
<html>
<head><title>Sample News</title></head>
<body>
USA is a country. US is also used to mean the same thing.
I was wondering if anyone out there could enlighten me on this car.
Contact us at admin@test.com or visit https://example.com !
The price is $45.55 on 01/02/06.
LOL!!! This movie is soooo goooood haha :)
a an the is at on in by to I it
</body>
</html>
"""

print(text)


# ===============================
# 2. HTML 태그 제거
# ===============================
print("\n" + "=" * 60)
print("2. HTML 태그 제거")
print("=" * 60)

clean_text = re.sub(r'<[^>]+>', '', text)
print(clean_text)


# ===============================
# 3. 정규화: 표기가 다른 단어 통합
# 예: USA, US -> usa
# ===============================
print("\n" + "=" * 60)
print("3. 표기가 다른 단어 통합")
print("=" * 60)

normalized_text = re.sub(r'\b(USA|US)\b', 'usa', clean_text, flags=re.IGNORECASE)
print(normalized_text)


# ===============================
# 4. 대소문자 통합
# 영어 NLP에서는 보통 소문자 변환을 많이 사용
# ===============================
print("\n" + "=" * 60)
print("4. 소문자 변환")
print("=" * 60)

normalized_text = normalized_text.lower()
print(normalized_text)


# ===============================
# 5. 이메일, URL 제거
# ===============================
print("\n" + "=" * 60)
print("5. 이메일 / URL 제거")
print("=" * 60)

normalized_text = re.sub(r'\S+@\S+', '', normalized_text)          # 이메일 제거
normalized_text = re.sub(r'https?://\S+|www\.\S+', '', normalized_text)  # URL 제거
print(normalized_text)


# ===============================
# 6. 숫자/통화/날짜 패턴 확인 또는 제거
# 필요에 따라 제거 가능
# ===============================
print("\n" + "=" * 60)
print("6. 숫자/특수 패턴 제거")
print("=" * 60)

# 달러 금액, 날짜, 숫자 제거 예시
normalized_text = re.sub(r'\$\d+(\.\d+)?', ' ', normalized_text)   # 금액 제거
normalized_text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', ' ', normalized_text)  # 날짜 제거
normalized_text = re.sub(r'\b\d+\b', ' ', normalized_text)         # 일반 숫자 제거
print(normalized_text)


# ===============================
# 7. 특수문자 정리
# 알파벳/공백 제외 문자 제거
# ===============================
print("\n" + "=" * 60)
print("7. 특수문자 제거")
print("=" * 60)

normalized_text = re.sub(r'[^a-z\s]', ' ', normalized_text)
normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
print(normalized_text)


# ===============================
# 8. 토큰화
# ===============================
print("\n" + "=" * 60)
print("8. 토큰화")
print("=" * 60)

tokens = normalized_text.split()
print(tokens)


# ===============================
# 9. 길이가 짧은 단어 제거
# 길이 1~2 제거
# 영어에서는 어느 정도 효과적
# ===============================
print("\n" + "=" * 60)
print("9. 길이가 짧은 단어 제거")
print("=" * 60)

filtered_tokens = [word for word in tokens if len(word) > 2]
print(filtered_tokens)


# ===============================
# 10. 등장 빈도 계산
# 너무 적게 등장하는 단어 확인
# ===============================
print("\n" + "=" * 60)
print("10. 단어 빈도수 계산")
print("=" * 60)

word_freq = Counter(filtered_tokens)
print(word_freq)


# ===============================
# 11. 등장 빈도가 1회 이하인 단어 제거 예시
# 실제 데이터가 많을 때 효과적
# ===============================
print("\n" + "=" * 60)
print("11. 희소 단어 제거 (빈도 1 이하 제거)")
print("=" * 60)

freq_filtered_tokens = [word for word in filtered_tokens if word_freq[word] > 1]
print(freq_filtered_tokens)


# ===============================
# 12. 최종 결과
# ===============================
print("\n" + "=" * 60)
print("12. 최종 전처리 결과")
print("=" * 60)
final_text = ' '.join(freq_filtered_tokens)
print(freq_filtered_tokens)
print(set(freq_filtered_tokens))
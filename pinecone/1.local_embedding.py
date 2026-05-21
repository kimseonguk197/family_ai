

from sentence_transformers import SentenceTransformer
import numpy as np

# 임베딩 모델 로드
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# A 주제: Pinecone 운용/인덱스/메타데이터
A_DOCS = [
	"Pinecone 인덱스는 dimension과 metric을 먼저 결정해서 생성한다.",
	"Pinecone upsert는 id 기준으로 벡터를 삽입 또는 갱신한다.",
	"metadata는 source, category 같은 필드로 필터링할 때 사용한다.",
	"namespace는 같은 인덱스 내부에서 논리적으로 데이터를 분리한다.",
]

# B 주제: 청킹/컨텍스트 구성/문서 전처리
B_DOCS = [
	"문서 청킹은 긴 문서를 작은 덩어리로 나눠 검색 품질을 높인다.",
	"overlap을 주면 청크 경계에서 문맥이 끊기는 문제를 줄일 수 있다.",
	"chunk가 너무 작으면 문맥이 부족하고, 너무 크면 검색이 뭉개질 수 있다.",
	"검색된 청크를 이어붙여 컨텍스트를 만들고 프롬프트에 넣는다.",
]

# 두 주제를 섞음
texts = A_DOCS + B_DOCS

# 문서 전체를 벡터로 변환 + 정규화 (cosine 유사도 계산 준비)
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.asarray(embeddings, dtype=np.float32)
print("embeddings:", embeddings.shape)  # (n, dim)

# 테스트용 질문
queries = {
	"A": "Pinecone에서 namespace와 metadata는 왜 쓰나요?",
	"B": "문서 청킹에서 overlap을 주는 이유가 뭔가요?",
}

def top_k(query: str, k: int = 3):
	# 질문을 동일하게 벡터로 변환
	q = model.encode([query], normalize_embeddings=True)
	q = np.asarray(q, dtype=np.float32)
	
	# 모든 문서와 유사도 계산 (cosine 유사도)
	scores = (embeddings @ q.T).reshape(-1)
	# 유사도 높은 순으로 top-k 선택
	idx = np.argsort(-scores)[:k]
	return [(i, float(scores[i]), texts[i]) for i in idx]

# 각 질문에 대해 가장 유사한 문서 출력
for label, q in queries.items():
	print("\n== Query", label, "==")
	print(q)
	for i, s, t in top_k(q, k=3):
		print(f"- score={s:.4f} | {t}")
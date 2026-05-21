import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# 0) 준비: 환경변수
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
	raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")

index_name = "pinecone-bradkim-practice"  # 앞 실습에서 만든 인덱스 사용
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# 1) 임베딩 모델
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2) A/B 주제 문서(업서트 대상)
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

# 모든 문장에 각각 A와 B로 토픽 지정
docs = [("A", i, t) for i, t in enumerate(A_DOCS)] + [("B", i, t) for i, t in enumerate(B_DOCS)]
texts = [t for _, _, t in docs]

# 3) 임베딩 생성
embs = model.encode(texts, normalize_embeddings=True)

# 4) upsert vectors 구성 : id(직접생성), metadata등을 저장
vectors = []
for (topic, i, text), vec in zip(docs, embs):
	vectors.append(
		{
			"id": f"{topic}:{i}",
			"values": vec.tolist(),
			"metadata": {
				"topic": topic,
				"text": text,
			},
		}
	)

# 5) upsert (namespace는 선택)
index.upsert(vectors=vectors, namespace="ab_test")

# 6) Query 함수 : 검색할 때 metadata 기준으로 필터 가능
def run_query(question: str, top_k: int = 3, topic_filter: str | None = None):
	q = model.encode([question], normalize_embeddings=True)[0].tolist()
	f = {"topic": {"$eq": topic_filter}} if topic_filter else None
	res = index.query(
		vector=q,
		top_k=top_k,
		namespace="ab_test",
		include_metadata=True,
		filter=f,
	)
	return res

# 7) A/B 대조 질문
questions = {
	"A": "Pinecone에서 namespace와 metadata는 왜 쓰나요?",
	"B": "문서 청킹에서 overlap을 주는 이유가 뭔가요?",
}

for label, q in questions.items():
	print("\n== Query", label, "==")
	print(q)
	res = run_query(q, top_k=3)
	for m in res["matches"]:
		print(f"- id={m['id']} score={m['score']:.4f} topic={m['metadata'].get('topic')} text={m['metadata'].get('text')}")

# 8) (선택) 필터로 의도적으로 반대 주제만 검색해보기
print("\n== Query A with topic=B filter (대조 실험) ==")	
res = run_query(questions["A"], top_k=3, topic_filter="B")
for m in res["matches"]:
	print(f"- id={m['id']} score={m['score']:.4f} topic={m['metadata'].get('topic')} text={m['metadata'].get('text')}")
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")

# 1) 설정값
INDEX_NAME = "pinecone-bradkim-practice"
NAMESPACE = "ab_test"
TOP_K = 3

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # (dim=384)
CHAT_MODEL = "gpt-4o-mini"  # 비용/속도 균형용 예시

# 2) 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embedder = SentenceTransformer(EMBED_MODEL)

# 3) 검색 함수 (Pinecone)
def retrieve(question: str, top_k: int = TOP_K, topic_filter: str | None = None) -> List[dict]:
    q = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    f = {"topic": {"$eq": topic_filter}} if topic_filter else None

    res = index.query(
        vector=q,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
        filter=f,
    )

    matches = res.get("matches", [])
    return matches

# 4) 컨텍스트 구성
def build_context(matches: List[dict]) -> str:
    # 점수/원문을 같이 넣어주면 디버깅에 좋음
    chunks = []
    for m in matches:
        meta = m.get("metadata") or {}
        text = meta.get("text") or ""
        chunks.append(f"- (score={m.get('score', 0):.4f}, id={m.get('id')}) {text}")

    context = "\n".join(chunks)
    return context

# 5) GPT 응답 생성
def generate_answer(question: str, context: str) -> str:
    system_msg = (
        "너는 RAG 기반 QA 어시스턴트야. "
        "반드시 제공된 CONTEXT 안에서 답해"
    )

    user_msg = f"""\
[QUESTION]\n{question}\n\n[CONTEXT]\n{context}\n\n[INSTRUCTIONS]\n- CONTEXT에 있는 내용만 이용해 한국어로 간결히 답해줘.\n"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content

# 6) 실행
if __name__ == "__main__":
    question = "Pinecone에서 namespace와 metadata는 왜 쓰나요?"

    matches = retrieve(question, top_k=TOP_K)
    context = build_context(matches)

    print("\n=== Retrieved Context ===")
    print(context)

    answer = generate_answer(question, context)
    print("\n=== Answer ===")
    print(answer)
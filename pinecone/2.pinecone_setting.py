import os
from pinecone import Pinecone, ServerlessSpec

# 1) 환경변수에 키를 넣어두는 방식 권장
# export PINECONE_API_KEY="..." (macOS/Linux)
# setx PINECONE_API_KEY "..." (Windows)
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")

pc = Pinecone(api_key=api_key)

index_name = "pinecone-bradkim-practice"
dimension = 384  # ← 실습 2에서 print(embeddings.shape)로 확인한 dim
metric = "cosine"  # normalize_embeddings=True면 cosine/dotproduct 둘 다 흔히 사용

# 2) 이미 있으면 만들지 않음
existing = [i["name"] for i in pc.list_indexes()]
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

# 3) 준비될 때까지 대기
pc.describe_index(index_name)

# 4) index 핸들 얻기 (이걸로 upsert/query)
index = pc.Index(index_name)

# 연결 확인용: 통계 출력(비어있으면 vector_count=0)
print(index.describe_index_stats())
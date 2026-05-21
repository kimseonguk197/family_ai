import os

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# 0) 환경변수
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not OPENAI_API_KEY:
	raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
if not PINECONE_API_KEY:
	raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")

# 1) Pinecone 설정
INDEX_NAME = "pinecone-bradkim-practice"
NAMESPACE = "ab_test"

# 2) 기존 sentence-transformers 모델과 동일하게 맞춤
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},  # 기존 코드와 동일
)


# 3) VectorStore 연결
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)


# 4) Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5) 테스트
question = "Pinecone에서 namespace와 metadata는 왜 쓰나요?"
docs = retriever.invoke(question) 

print("retrieved:", len(docs))
for i, d in enumerate(docs, start=1):
    print(f"\n--- chunk {i} ---")
    print(d.page_content)
    print("metadata:", d.metadata)
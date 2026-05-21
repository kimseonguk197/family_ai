import os

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")

INDEX_NAME = "pinecone-bradkim-practice"
NAMESPACE = "ab_test"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

texts = [
    "Pinecone 인덱스는 dimension과 metric을 먼저 결정해서 생성한다.",
    "Pinecone upsert는 id 기준으로 벡터를 삽입 또는 갱신한다.",
    "metadata는 source, category 같은 필드로 필터링할 때 사용한다.",
    "namespace는 같은 인덱스 내부에서 논리적으로 데이터를 분리한다.",
    "문서 청킹은 긴 문서를 작은 덩어리로 나눠 검색 품질을 높인다.",
    "overlap을 주면 청크 경계에서 문맥이 끊기는 문제를 줄일 수 있다.",
    "chunk가 너무 작으면 문맥이 부족하고, 너무 크면 검색이 뭉개질 수 있다.",
    "검색된 청크를 이어붙여 컨텍스트를 만들고 프롬프트에 넣는다.",
]

metadatas = [
    {"topic": "A", "text": texts[0]},
    {"topic": "A", "text": texts[1]},
    {"topic": "A", "text": texts[2]},
    {"topic": "A", "text": texts[3]},
    {"topic": "B", "text": texts[4]},
    {"topic": "B", "text": texts[5]},
    {"topic": "B", "text": texts[6]},
    {"topic": "B", "text": texts[7]},
]

ids = ["A:0", "A:1", "A:2", "A:3", "B:0", "B:1", "B:2", "B:3"]

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)

vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
print("upserted:", len(texts))
import os

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

# 2) LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)

# 3) Embeddings + VectorStore + Retriever
# 기존 업서트/조회 코드와 동일한 임베딩 모델/정규화 설정으로 맞춤
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4) 컨텍스트 포맷 함수
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 5) Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 RAG 기반 QA 어시스턴트야. "
            "반드시 제공된 CONTEXT 안에서 답해",
        ),
        (
            "user",
            "[QUESTION]\n{question}\n\n[CONTEXT]\n{context}\n\n[INSTRUCTIONS]\n- CONTEXT에 있는 내용만 이용해 한국어로 간결히 답해줘.\n",
        ),
    ]
)

# 6) LCEL 체인 : "질문 → 문서 검색 → prompt 생성 → LLM 호출 → 문자열 출력"
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = "Pinecone에서 namespace와 metadata는 왜 쓰나요?"
answer = chain.invoke(question)
print(answer)
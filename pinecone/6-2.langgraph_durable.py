import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# [NEW] PostgreSQL 체크포인터 추가
from langgraph.checkpoint.postgres import PostgresSaver

# 1. 환경 변수 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. 터미널에서 export/set 명령어로 설정하세요.")

llm = ChatOpenAI(model="gpt-4o-mini")

# [NEW] DB 연결 문자열을 코드에 직접 작성
DB_URI = "postgresql://postgres:1234@localhost:5432/langgraph_db"

# 2. 상태(State) 정의: 그래프 전체에서 노드 간에 전달되는 '공유 메모리' 역할
class AgentState(TypedDict):
    topic: str           # 글의 주제
    content: str         # AI가 작성한 글 본문 (노드를 거치며 업데이트됨)
    revision_count: int  # 무한 루프 방지를 위한 카운터 (수정 횟수)
    is_good: bool        # 검토 노드(critic)에서 판단한 품질 통과 여부


# 3. 노드(Node) 정의: 각 단계에서 실행될 구체적인 동작(Action)
def writer(state: AgentState):
    """
    - 처음 진입 시: 사용자의 주제를 바탕으로 초안을 작성함
    - 다시 돌아올 시: 이전의 부족한 부분을 보완하여 '개선된 버전'으로 덮어씀
    """
    print(f"--- [Writer Node] 글 작성 중 (현재 수정 횟수: {state['revision_count']}) ---")
    prompt = f"{state['topic']}에 대해 전문적인 블로그 글을 써줘. 현재 수정 순번: {state['revision_count']}"
    response = llm.invoke(prompt)

    # 수정 횟수를 1 증가시켜 상태를 업데이트함 (다음 루프의 종료 조건으로 활용)
    return {
        "content": response.content,
        "revision_count": state["revision_count"] + 1
    }


def critic(state: AgentState):
    """
    - Writer가 작성한 결과물(content)을 읽어서 '통과/탈락'을 결정함
    - 여기서는 단순 글자 수로 판단하지만, 실제로는 다른 LLM이 평가하게 설계 가능
    """
    print("--- Critic Node에서 품질 검토 중 ---")
    # 비즈니스 로직: 글자 수가 200자 미만이면 '부족'으로 판단하여 재작성 지시 예정
    is_good = len(state["content"]) > 200
    return {"is_good": is_good}


# 4. 조건부 엣지(Conditional Edge): 흐름의 분기점 제어
def should_continue(state: AgentState):
    """
    1. 'end'로 가는 경우:
       - 품질이 만족스럽거나(is_good=True)
       - 품질은 안 좋아도 너무 많이 고쳤을 때(revision_count > 2)
    2. 'rewrite'로 가는 경우: 품질이 미달인데 아직 수정 기회가 남았을 때 -> 다시 writer_node로 회귀
    """
    if state["is_good"] or state["revision_count"] > 2:
        print("--- 만족 또는 횟수 초과로 인해 프로세스 종료 ---")
        return "end"

    print("--- 품질 미달로 인한 재작성(Cycle) 발생 ---")
    return "rewrite"


# 5. 그래프 빌드 및 워크플로우 설계
workflow = StateGraph(AgentState)

# 각 노드를 그래프에 등록
workflow.add_node("writer_node", writer)
workflow.add_node("critic_node", critic)

# 흐름 정의
workflow.set_entry_point("writer_node")              # 시작점: 무조건 글부터 써야 함
workflow.add_edge("writer_node", "critic_node")     # 작성 후엔 반드시 검토를 거침

# critic_node가 끝나면 'should_continue' 함수를 실행해 다음 행방을 결정함
workflow.add_conditional_edges(
    "critic_node",
    should_continue,
    {
        "end": END,                 # 'end' 반환 시 프로세스 종료
        "rewrite": "writer_node"    # 'rewrite' 반환 시 writer_node로 되돌아감 (Cycle 형성)
    }
)

# [NEW] PostgreSQL checkpointer 생성
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # [NEW] 최초 1회 체크포인트 테이블 생성
    checkpointer.setup()

    # [CHANGED] checkpointer를 연결해서 compile
    app = workflow.compile(checkpointer=checkpointer)

    # 6. 실행
    inputs = {
        "topic": "LangGraph의 장점을 짧게 2줄로 대답해",
        "content": "",         # [NEW] 초기 상태값 명시
        "revision_count": 0,
        "is_good": False       # [NEW] 초기 상태값 명시
    }

    # [NEW] thread_id 지정
    # 같은 thread_id를 쓰면 같은 요청 흐름의 checkpoint가 누적됨
    config = {
        "configurable": {
            "thread_id": "user-001-blog-request"
        }
    }

    final_state = app.invoke(inputs, config=config)

    print("\n" + "=" * 50)
    print(f"최종 결과물 (총 수정 횟수: {final_state['revision_count']})")
    print(final_state["content"])

    # [NEW] 현재 thread의 최신 checkpoint 상태 조회
    snapshot = app.get_state(config)
    print("\n[최신 체크포인트 상태]")
    print(snapshot.values)
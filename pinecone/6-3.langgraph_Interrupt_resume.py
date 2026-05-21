import os
from pathlib import Path
from typing import TypedDict, List, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
import time
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI()

# 1. DB 연결
DB_URI = os.environ.get(
    "LANGGRAPH_DB_URI",
    "postgresql://postgres:1234@localhost:5432/langgraph_db"
)

THREAD_ID = "simple-resume-demo-001"


# 2. 상태 정의
class State(TypedDict):
    files: List[str]
    current_index: int
    summaries: List[str]
    progress_summary: str
    final_summary: str
    run_mode: Literal["first", "resume"]


# 3. 첫 실행용 노드: 문서 2개까지만 처리
def read_first_two_docs(state: State):
    files = state["files"]
    current_index = state["current_index"]
    summaries = list(state["summaries"])

    stop_index = min(current_index + 2, len(files))

    for i in range(current_index, stop_index):
        text = Path(files[i]).read_text(encoding="utf-8")

        # 🔥 GPT 요약
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "주어진 문서를 2줄로 핵심만 요약해라."},
                {"role": "user", "content": text}
            ]
        )

        summary = response.choices[0].message.content.strip()
        one_line_summary = f"[{Path(files[i]).name}] {summary}"

        summaries.append(one_line_summary)

    return {
        "current_index": stop_index,
        "summaries": summaries,
    }


# 4. 중간 통합요약 생성
def make_progress_summary(state: State):
    summaries = state["summaries"]
    current_index = state["current_index"]

    if not summaries:
        return {"progress_summary": "아직 처리된 문서가 없습니다."}

    joined = "\n".join(summaries)
    progress = f"""중간 통합 요약

현재까지 처리한 문서 수: {current_index}
전체 문서 수: {len(state['files'])}

현재까지의 개별 요약:
{joined}
"""
    return {"progress_summary": progress}


# 5. 재실행용 노드: 남은 문서만 처리
def read_remaining_docs(state: State):
    files = state["files"]
    current_index = state["current_index"]
    summaries = list(state["summaries"])

    for i in range(current_index, len(files)):
        text = Path(files[i]).read_text(encoding="utf-8")

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "주어진 문서를 2줄로 핵심만 요약해라."},
                {"role": "user", "content": text}
            ]
        )

        summary = response.choices[0].message.content.strip()
        one_line_summary = f"[{Path(files[i]).name}] {summary}"

        summaries.append(one_line_summary)

    return {
        "current_index": len(files),
        "summaries": summaries,
    }


# 6. 최종 통합요약 생성
def make_final_summary(state: State):
    summaries = state["summaries"]

    if not summaries:
        return {"final_summary": "처리된 문서가 없습니다."}

    joined = "\n".join(summaries)
    final = f"""최종 통합 요약

총 문서 수: {len(state['files'])}
처리 완료 수: {state['current_index']}

전체 개별 요약:
{joined}
"""
    return {"final_summary": final}


# 7. 시작 분기
def route_by_mode(state: State):
    if state["run_mode"] == "first":
        return "read_first_two_docs"
    return "read_remaining_docs"


# 8. 그래프 정의
workflow = StateGraph(State)

workflow.add_node("read_first_two_docs", read_first_two_docs)
workflow.add_node("make_progress_summary", make_progress_summary)
workflow.add_node("read_remaining_docs", read_remaining_docs)
workflow.add_node("make_final_summary", make_final_summary)

workflow.add_conditional_edges(
    START,
    route_by_mode,
    {
        "read_first_two_docs": "read_first_two_docs",
        "read_remaining_docs": "read_remaining_docs",
    }
)

# 첫 실행: 2개 처리 -> 중간 통합요약 -> 종료
workflow.add_edge("read_first_two_docs", "make_progress_summary")
workflow.add_edge("make_progress_summary", END)

# 재실행: 남은 문서 처리 -> 최종 통합요약 -> 종료
workflow.add_edge("read_remaining_docs", "make_final_summary")
workflow.add_edge("make_final_summary", END)


def main():
    start_time = time.time()  # 시작 시간
    files = [
        "docs/doc1.txt",
        "docs/doc2.txt",
        "docs/doc3.txt"
    ]

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        app = workflow.compile(checkpointer=checkpointer)

        snapshot = app.get_state(config)

        has_saved_state = (
            snapshot is not None
            and snapshot.values is not None
            and len(snapshot.values) > 0
        )

        # 1. 저장된 상태가 없으면 첫 실행
        if not has_saved_state:
            print("=== 첫 실행 ===")
            print("DB에 저장된 상태가 없어서 doc1~doc2까지만 처리합니다.\n")

            initial_state = {
                "files": files,
                "current_index": 0,
                "summaries": [],
                "progress_summary": "",
                "final_summary": "",
                "run_mode": "first"
            }

            result = app.invoke(initial_state, config=config)
            print("[1차 실행 결과]")
            print(result)

            saved_snapshot = app.get_state(config)
            print("\n[DB에 저장된 상태]")
            print(saved_snapshot.values)

            print("\n=== 중간 통합요약 ===")
            print(saved_snapshot.values["progress_summary"])

            print("\n프로그램을 종료합니다.")
            end_time = time.time()  # 종료 시간
            print(f"\n⏱ 실행 시간: {end_time - start_time:.4f}초")
            return

        # 2. 저장된 상태가 있으면 이어서 실행
        saved_values = snapshot.values
        saved_files = saved_values.get("files", files)
        current_index = saved_values.get("current_index", 0)
        summaries = saved_values.get("summaries", [])
        progress_summary = saved_values.get("progress_summary", "")
        final_summary = saved_values.get("final_summary", "")

        print("=== 재실행 ===")
        print(f"DB에 저장된 상태를 발견했습니다. current_index={current_index}\n")

        # 이미 끝난 상태면 재요약하지 않음
        if current_index >= len(saved_files):
            print("[이미 모든 문서 처리가 완료된 상태입니다.]")
            print(saved_values)

            if progress_summary:
                print("\n=== 저장된 중간 통합요약 ===")
                print(progress_summary)

            if final_summary:
                print("\n=== 저장된 최종 통합요약 ===")
                print(final_summary)
            end_time = time.time()  # 종료 시간
            print(f"\n⏱ 실행 시간: {end_time - start_time:.4f}초")
            return

        resumed_state = {
            "files": saved_files,
            "current_index": current_index,
            "summaries": summaries,
            "progress_summary": progress_summary,
            "final_summary": final_summary,
            "run_mode": "resume"
        }

        result = app.invoke(resumed_state, config=config)
        print("[재실행 결과]")
        print(result)

        final_snapshot = app.get_state(config)
        print("\n[최종 저장 상태]")
        print(final_snapshot.values)

        if final_snapshot.values.get("progress_summary"):
            print("\n=== 기존 중간 통합요약 ===")
            print(final_snapshot.values["progress_summary"])

        print("\n=== 최종 통합요약 ===")
        print(final_snapshot.values["final_summary"])

    end_time = time.time()  # 종료 시간
    print(f"\n⏱ 실행 시간: {end_time - start_time:.4f}초")

if __name__ == "__main__":
    main()
"""跑一遍 Agentic RAG：对比中海地产和四川发展的短期借款。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from main_graph import build_app
from state import initial_state


QUESTION = "对比一下中海地产和四川发展的短期借款。哪家公司的借款更多？"
CONFIG = {"configurable": {"thread_id": "test-1"}, "recursion_limit": 15}


def load_api_key() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            f"未读到 DEEPSEEK_API_KEY。请在 {env_path} 写入有效密钥后再运行。"
        )


def child_preview(doc: dict, limit: int = 120) -> str:
    children = doc.get("children") or []
    if not children:
        return ""
    text = " ".join(str(children[0].get("text") or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def print_event(node: str, update: dict) -> str:
    print(f"\n[{node}]")
    if node == "planner":
        tasks = update.get("sub_tasks") or []
        print(f"子任务 {len(tasks)} 条：")
        for index, task in enumerate(tasks, start=1):
            print(f"  {index}. {task}")
        print(f"当前任务：{update.get('current_task', '')}")
    elif node == "retriever":
        docs = update.get("retrieved_docs") or []
        if not docs:
            print("没有检索到页面")
            return ""
        latest = max(int(doc.get("task_index", 0)) for doc in docs)
        fresh = [doc for doc in docs if int(doc.get("task_index", 0)) == latest]
        print(f"当前子任务：{fresh[0].get('task', '') if fresh else ''}")
        for doc in fresh:
            meta = doc.get("metadata") or {}
            print(
                f"  页面 {meta.get('page_num')}  "
                f"{meta.get('company_name')}  "
                f"parent_id={doc.get('parent_id')}"
            )
            preview = child_preview(doc)
            if preview:
                print(f"    {preview}")
    elif node == "grader":
        print(f"评估：{update.get('grade', '')}")
    elif node == "rewrite":
        print(f"改写检索词：{update.get('current_task', '')}")
        print(f"loop_count：{update.get('loop_count', '')}")
    elif node == "advance":
        print(f"进入下一条：{update.get('current_task', '')}")
    elif node == "generator":
        print("已生成最终答案")
        return str(update.get("final_answer") or "")
    return ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_api_key()
    app = build_app()
    final_answer = ""
    print(f"问题：{QUESTION}")
    for event in app.stream(initial_state(QUESTION), CONFIG, stream_mode="updates"):
        for node, update in event.items():
            answer = print_event(node, update or {})
            if answer:
                final_answer = answer
    print("\n========== 最终答案 ==========")
    print(final_answer or "generator 没有返回 final_answer")


if __name__ == "__main__":
    main()

"""Agentic RAG 的状态定义。

main_graph.py 用 StateGraph(AgentState) 装载这些字段。
messages 使用 add_messages，节点只返回新消息，图会把它追加到历史里。
其余字段由节点整段覆盖。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChildHit(TypedDict):
    """一次检索命中的子块。向量库里只有子块。"""

    child_id: str
    text: str
    distance: float
    block_type: str


class RetrievedDoc(TypedDict):
    """一个父块及其命中的子块。生成和评估读的是父块全文。"""

    task: str
    task_index: int
    parent_id: str
    parent_text: str
    metadata: dict[str, Any]
    children: list[ChildHit]


class AgentState(TypedDict):
    """整张状态机共享的一份状态。

    grade 和 task_index 是条件边要读的信号：
    grade 表示当前子任务的检索是否够用，task_index 指向 sub_tasks 里正在做的那一条。
    """

    original_question: str
    sub_tasks: list[str]
    current_task: str
    retrieved_docs: list[RetrievedDoc]
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int
    final_answer: str
    grade: str
    task_index: int


def initial_state(question: str) -> AgentState:
    """图的起点。planner_node 负责填 sub_tasks 和 current_task。"""
    return {
        "original_question": question.strip(),
        "sub_tasks": [],
        "current_task": "",
        "retrieved_docs": [],
        "messages": [],
        "loop_count": 0,
        "final_answer": "",
        "grade": "",
        "task_index": 0,
    }

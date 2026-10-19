"""把六个节点组装成可编译的 StateGraph。

测试入口稍后单独写。导入本模块不会打开向量库，也不会请求模型。
调用 build_app() 时才创建 ChatDeepSeek、载入 Chroma，并编译图。
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nodes import ParentRetriever, build_llm, build_nodes, open_retriever, route_after_grade
from state import AgentState


def build_app(
    llm: BaseChatModel | None = None,
    retriever: ParentRetriever | None = None,
) -> CompiledStateGraph:
    """组装并编译图。llm 或 retriever 为空时使用 DeepSeek 和本地向量库。"""
    if llm is None:
        llm = build_llm()
    if retriever is None:
        retriever = open_retriever()

    node_fns = build_nodes(llm=llm, retriever=retriever)
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", node_fns["planner"])
    workflow.add_node("retriever", node_fns["retriever"])
    workflow.add_node("grader", node_fns["grader"])
    workflow.add_node("rewrite", node_fns["rewrite"])
    workflow.add_node("advance", node_fns["advance"])
    workflow.add_node("generator", node_fns["generator"])

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "grader")
    workflow.add_edge("rewrite", "retriever")
    workflow.add_edge("advance", "retriever")
    workflow.add_edge("generator", END)

    # route_after_grade 读取 grade、loop_count、task_index，
    # 返回 rewrite、advance 或 generator。
    workflow.add_conditional_edges(
        "grader",
        route_after_grade,
        {
            "rewrite": "rewrite",
            "advance": "advance",
            "generator": "generator",
        },
    )

    app = workflow.compile(checkpointer=MemorySaver())
    return app

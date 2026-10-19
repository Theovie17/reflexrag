"""Agentic RAG 的节点。

固定边和条件边在 main_graph.py 里组装。走向如下：

    START -> planner -> retriever -> grader
    rewrite -> retriever
    advance -> retriever
    generator -> END

    grader
      ├─ grade == insufficient 且 loop_count < MAX_LOOPS -> rewrite
      ├─ 后面还有子任务 -> advance
      └─ 已经是最后一条子任务 -> generator

planner、grader、rewrite、generator 都调用注入的聊天模型。
build_llm() 使用 langchain_deepseek.ChatDeepSeek，模型名 deepseek-flash，
密钥只从环境变量 DEEPSEEK_API_KEY 读取。思考模式默认开启，这里关掉，
以便直接拿到 JSON、评估词和检索问句。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from state import AgentState, ChildHit, RetrievedDoc


MAX_LOOPS = 2
TOP_K = 3
GRADE_SUFFICIENT = "sufficient"
GRADE_INSUFFICIENT = "insufficient"
DEFAULT_CHAT_MODEL = "deepseek-flash"

PLANNER_SYSTEM = (
    "你是债券财报问答的任务拆解器。"
    "把用户问题拆成可以分别检索的子任务。"
    "问题只问一个公司的一个指标时，数组里只放原问题。"
    "比较两家公司或两个指标时，拆成多条，每条都保留公司名和科目名。"
    "只输出 JSON 字符串数组，不要解释。"
)
GRADER_SYSTEM = (
    "你负责判断检索材料能否回答当前子任务。"
    "公司、科目和问到的数字都在材料里时，只输出 sufficient。"
    "缺少关键事实时，只输出 insufficient。"
    "不要输出别的文字。"
)
REWRITE_SYSTEM = (
    "上一轮检索没有回答当前子任务。请改写成一条更具体的中文检索问句。"
    "保留公司名和会计科目，补上报表名称，例如资产负债表。"
    "新问句必须和原检索词不同。只输出这一条问句。"
)
GENERATOR_SYSTEM = (
    "你是债券半年度报告问答助手。只根据给定材料回答原问题。"
    "写明公司、报告期、页码和数字。材料里没有的数字不要编造。"
    "多条子任务的材料都要考虑到。用中文回答。"
)


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content)


def build_llm(
    model: str = DEFAULT_CHAT_MODEL,
    *,
    api_key: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """创建 ChatDeepSeek。不在代码里写密钥。

    官方当前模型名是 deepseek-flash。reasoning_effort="none" 且
    extra_body 里的 thinking.type=disabled 都表示关闭思考模式。
    非思考模式才接受 temperature。
    """
    from langchain_deepseek import ChatDeepSeek

    key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")
    return ChatDeepSeek(
        model=model,
        api_key=key,
        temperature=temperature,
        max_retries=2,
        reasoning_effort="none",
        extra_body={"thinking": {"type": "disabled"}},
    )


def call_llm(llm: BaseChatModel, system: str, user: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = message_text(response).strip()
    if not text:
        raise RuntimeError("模型返回了空正文。请确认思考模式已关闭。")
    return text


def parse_task_list(text: str) -> list[str] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    found = re.search(r"\[[\s\S]*\]", cleaned)
    if found:
        candidates.append(found.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        tasks = [str(item).strip() for item in data if str(item).strip()]
        if tasks:
            return tasks
    return None


def parse_grade(text: str) -> str | None:
    token = re.split(r"[\s,，。:：]+", text.strip().lower())[0].strip("`\"'")
    if token.startswith(("insufficient", "不能", "不够", "不足")):
        return GRADE_INSUFFICIENT
    if token.startswith(("sufficient", "可以", "够")) or token == "能":
        return GRADE_SUFFICIENT
    return None


def clean_query(text: str) -> str:
    line = text.strip().splitlines()[0].strip()
    line = line.strip("`\"'“”")
    line = re.sub(r"^(检索词|新检索词|问句)[:：]\s*", "", line)
    return line.strip()


def docs_for_task(state: AgentState) -> list[RetrievedDoc]:
    index = state.get("task_index", 0)
    return [doc for doc in state.get("retrieved_docs", []) if doc.get("task_index") == index]


def doc_blob(docs: list[RetrievedDoc]) -> str:
    parts: list[str] = []
    for doc in docs:
        company = doc["metadata"].get("company_name", "")
        page = doc["metadata"].get("page_num", "")
        parts.append(f"{company} 第{page}页 {doc['parent_id']}")
        parts.append(doc.get("parent_text") or "")
        for child in doc.get("children", []):
            parts.append(child.get("text") or "")
    return "\n".join(parts)


def preview_docs(docs: list[RetrievedDoc], limit: int = 800) -> str:
    text = doc_blob(docs)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def plan_tasks(llm: BaseChatModel, question: str) -> list[str]:
    raw = call_llm(llm, PLANNER_SYSTEM, question)
    tasks = parse_task_list(raw)
    if tasks:
        return tasks
    raw = call_llm(
        llm,
        PLANNER_SYSTEM,
        f"用户问题：{question}\n\n上次输出无法解析：\n{raw}\n只重新输出 JSON 字符串数组。",
    )
    tasks = parse_task_list(raw)
    if not tasks:
        raise RuntimeError(f"任务拆解结果无法解析：{raw}")
    return tasks


def judge_task(llm: BaseChatModel, task: str, docs: list[RetrievedDoc]) -> str:
    prompt = f"子任务：{task}\n\n检索材料：\n{preview_docs(docs) or '（没有检索到材料）'}"
    raw = call_llm(llm, GRADER_SYSTEM, prompt)
    grade = parse_grade(raw)
    if grade:
        return grade
    raw = call_llm(llm, GRADER_SYSTEM, prompt + "\n\n只输出 sufficient 或 insufficient。")
    grade = parse_grade(raw)
    if not grade:
        raise RuntimeError(f"评估结果无法解析：{raw}")
    return grade


def rewrite_query(llm: BaseChatModel, task: str, loop_count: int, docs: list[RetrievedDoc]) -> str:
    prompt = (
        f"原检索词：{task}\n"
        f"已重试次数：{loop_count}\n"
        f"上一轮材料：\n{preview_docs(docs, limit=500) or '（没有检索到材料）'}"
    )
    rewritten = clean_query(call_llm(llm, REWRITE_SYSTEM, prompt))
    if rewritten and rewritten != task:
        return rewritten
    rewritten = clean_query(
        call_llm(llm, REWRITE_SYSTEM, prompt + "\n\n新问句必须和原检索词不同，只输出这一句。")
    )
    if not rewritten or rewritten == task:
        raise RuntimeError(f"改写结果无效：{rewritten}")
    return rewritten


class ParentRetriever:
    """搜子块，再按 parent_id 把父块全文从内存字典里取出来。"""

    def __init__(
        self,
        persist_dir: Path,
        parents_path: Path,
        cache_dir: Path,
        model_name: str,
        top_k: int = TOP_K,
    ) -> None:
        import chromadb
        from langchain_chroma import Chroma

        from build_index import COLLECTION_NAME, build_embeddings, load_parents

        self.top_k = top_k
        self.parents = load_parents(parents_path)
        embeddings = build_embeddings(model_name, cache_dir)
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            create_collection_if_not_exists=False,
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        hits = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for doc, distance in hits:
            raw = dict(doc.metadata or {})
            parent_id = str(raw.get("parent_id") or "")
            parent = self.parents.get(parent_id, {})
            parent_meta = parent.get("metadata") or {}
            if parent_id not in grouped:
                page_num = raw.get("page_num", parent_meta.get("page_num", 0))
                order.append(parent_id)
                grouped[parent_id] = {
                    "parent_id": parent_id,
                    "parent_text": str(parent.get("text") or ""),
                    "metadata": {
                        "parent_id": parent_id,
                        "source_file": str(raw.get("source_file") or parent_meta.get("source_file") or ""),
                        "page_num": int(page_num or 0),
                        "company_name": str(raw.get("company_name") or parent_meta.get("company_name") or ""),
                    },
                    "children": [],
                }
            child: ChildHit = {
                "child_id": str(raw.get("child_id") or ""),
                "text": doc.page_content,
                "distance": float(distance),
                "block_type": str(raw.get("block_type") or ""),
            }
            grouped[parent_id]["children"].append(child)
        for item in grouped.values():
            item["children"].sort(key=lambda child: child["distance"])
        return [grouped[parent_id] for parent_id in order]


def build_nodes(
    *,
    llm: BaseChatModel,
    retriever: ParentRetriever,
) -> dict[str, Callable[[AgentState], dict]]:
    """返回图上的六个节点。main_graph.py 按这些名字注册。"""

    def planner_node(state: AgentState) -> dict:
        """调用模型拆 original_question。简单问题得到一条子任务。

        出去之后固定进入 retriever。current_task 取第一条，loop_count 从 0 开始。
        """
        question = state["original_question"].strip()
        tasks = plan_tasks(llm, question)
        current = tasks[0]
        plan = "、".join(tasks)
        return {
            "sub_tasks": tasks,
            "current_task": current,
            "task_index": 0,
            "loop_count": 0,
            "grade": "",
            "retrieved_docs": [],
            "messages": [
                HumanMessage(content=question),
                AIMessage(content=f"拆成 {len(tasks)} 个子任务：{plan}"),
            ],
        }

    def retriever_node(state: AgentState) -> dict:
        """用 current_task 检索子块，并挂上同一个 parent_id 的父块全文。

        出去之后固定进入 grader。
        """
        task = state["current_task"].strip()
        index = int(state.get("task_index", 0))
        fresh: list[RetrievedDoc] = []
        for item in retriever.search(task):
            fresh.append(
                {
                    "task": task,
                    "task_index": index,
                    "parent_id": item["parent_id"],
                    "parent_text": item["parent_text"],
                    "metadata": item["metadata"],
                    "children": item["children"],
                }
            )
        kept = [doc for doc in state.get("retrieved_docs", []) if doc.get("task_index") != index]
        parent_ids = "、".join(doc["parent_id"] for doc in fresh) or "无"
        return {
            "retrieved_docs": kept + fresh,
            "messages": [AIMessage(content=f"子任务「{task}」检索到父块：{parent_ids}")],
        }

    def grader_node(state: AgentState) -> dict:
        """调用模型判断当前材料能否回答 current_task，把结论写入 grade。

        出去之后由 route_after_grade 选择 rewrite、advance 或 generator。
        """
        task = state["current_task"]
        grade = judge_task(llm, task, docs_for_task(state))
        return {
            "grade": grade,
            "messages": [AIMessage(content=f"评估「{task}」：{grade}")],
        }

    def rewrite_node(state: AgentState) -> dict:
        """调用模型根据当前任务和上一轮材料改写检索词。

        出去之后固定回到 retriever。loop_count 加 1，grade 清空。
        """
        task = state["current_task"]
        rewritten = rewrite_query(llm, task, int(state.get("loop_count", 0)), docs_for_task(state))
        return {
            "current_task": rewritten,
            "loop_count": int(state.get("loop_count", 0)) + 1,
            "grade": "",
            "messages": [AIMessage(content=f"改写检索词：{rewritten}")],
        }

    def advance_node(state: AgentState) -> dict:
        """当前子任务结束后切到下一条。出去之后固定回到 retriever。"""
        index = int(state.get("task_index", 0)) + 1
        tasks = state.get("sub_tasks") or []
        if index >= len(tasks):
            raise RuntimeError("advance_node 只应在还有剩余子任务时进入")
        nxt = tasks[index]
        return {
            "task_index": index,
            "current_task": nxt,
            "loop_count": 0,
            "grade": "",
            "messages": [AIMessage(content=f"进入下一子任务：{nxt}")],
        }

    def generator_node(state: AgentState) -> dict:
        """调用模型阅读原始问题和全部父块，写出 final_answer。出去之后是 END。"""
        question = state["original_question"]
        docs = state.get("retrieved_docs") or []
        answer = call_llm(
            llm,
            GENERATOR_SYSTEM,
            f"问题：{question}\n\n材料：\n{preview_docs(docs, limit=6000) or '（没有检索到材料）'}",
        )
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return {
        "planner": planner_node,
        "retriever": retriever_node,
        "grader": grader_node,
        "rewrite": rewrite_node,
        "advance": advance_node,
        "generator": generator_node,
    }


def route_after_grade(state: AgentState) -> str:
    """grader 的条件边。同时看 grade、loop_count 和 task_index。"""
    tasks = state.get("sub_tasks") or []
    has_next = int(state.get("task_index", 0)) + 1 < len(tasks)
    if state.get("grade") == GRADE_SUFFICIENT:
        return "advance" if has_next else "generator"
    if int(state.get("loop_count", 0)) < MAX_LOOPS:
        return "rewrite"
    return "advance" if has_next else "generator"


def open_retriever(
    script_dir: Path | None = None,
    *,
    model_name: str | None = None,
    top_k: int = TOP_K,
) -> ParentRetriever:
    """打开已经建好的 ./chroma_db，并载入 ./parents.json。"""
    from build_index import DEFAULT_MODEL

    root = script_dir or Path(__file__).resolve().parent
    return ParentRetriever(
        persist_dir=root / "chroma_db",
        parents_path=root / "parents.json",
        cache_dir=root / "models",
        model_name=model_name or DEFAULT_MODEL,
        top_k=top_k,
    )


__all__ = [
    "MAX_LOOPS",
    "DEFAULT_CHAT_MODEL",
    "ParentRetriever",
    "build_llm",
    "build_nodes",
    "open_retriever",
    "route_after_grade",
]

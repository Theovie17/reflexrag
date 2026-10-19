"""用 FastAPI 把 ReflexRAG 的 LangGraph 包成 SSE 接口。

POST /chat 的请求体：{"message": "...", "thread_id": "可选，默认 default"}
响应是 text/event-stream。每条事件一行 data，内容是 JSON。

事件类型：
- status：节点刚开始。text 例如「正在规划子任务」「正在检索：…」
- planner：拆解完成。sub_tasks 是子任务列表。
- retriever：本轮检索完成。pages 里是页码、公司和 parent_id，不含整页父块。
- grader：grade 为 sufficient 或 insufficient。
- rewrite：改写后的检索词和 loop_count。没改写时不会出现。
- advance：切到下一条子任务。
- answer：generator 的最终回答。
- done：这一轮结束。
- error：图运行失败。

同一个 thread_id 会复用 MemorySaver 里的对话。服务进程退出后记忆消失。
新问题会重新规划并检索；消息历史留在检查点里。
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from main_graph import build_app
from state import initial_state


ROOT = Path(__file__).resolve().parent
RECURSION_LIMIT = 30
NODE_START_TEXT = {
    "planner": "正在规划子任务",
    "retriever": "正在检索",
    "grader": "正在评估检索结果",
    "rewrite": "正在改写检索词",
    "advance": "正在进入下一子任务",
    "generator": "正在生成回答",
}

graph: Any = None
graph_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = "default"


def load_api_key() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError(f"未读到 DEEPSEEK_API_KEY。请在 {ROOT / '.env'} 写入有效密钥。")


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def child_preview(doc: dict[str, Any], limit: int = 160) -> str:
    children = doc.get("children") or []
    if not children:
        return ""
    text = " ".join(str(children[0].get("text") or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def fresh_pages(update: dict[str, Any]) -> list[dict[str, Any]]:
    """只取这一轮 task_index 的命中，避免把上一轮子任务的页面再推一次。"""
    docs = update.get("retrieved_docs") or []
    if not docs:
        return []
    latest = max(int(doc.get("task_index", 0)) for doc in docs)
    pages: list[dict[str, Any]] = []
    for doc in docs:
        if int(doc.get("task_index", 0)) != latest:
            continue
        meta = doc.get("metadata") or {}
        pages.append(
            {
                "task": doc.get("task") or "",
                "parent_id": doc.get("parent_id") or "",
                "page_num": meta.get("page_num"),
                "company_name": meta.get("company_name") or "",
                "source_file": meta.get("source_file") or "",
                "preview": child_preview(doc),
            }
        )
    return pages


def current_task(data: dict[str, Any]) -> str:
    raw = data.get("input")
    if isinstance(raw, dict):
        return str(raw.get("current_task") or "")
    return ""


def start_event(node: str, data: dict[str, Any]) -> dict[str, Any]:
    text = NODE_START_TEXT[node]
    task = current_task(data)
    if node == "retriever" and task:
        text = f"正在检索：{task}"
    elif node == "grader" and task:
        text = f"正在评估：{task}"
    return {"type": "status", "node": node, "text": text}


def end_event(node: str, output: dict[str, Any]) -> dict[str, Any] | None:
    if node == "planner":
        tasks = [str(task) for task in output.get("sub_tasks") or []]
        return {
            "type": "planner",
            "node": node,
            "text": f"拆解出 {len(tasks)} 个子任务",
            "sub_tasks": tasks,
            "current_task": output.get("current_task") or "",
        }
    if node == "retriever":
        pages = fresh_pages(output)
        if pages:
            labels = "、".join(f"{item['company_name']}第{item['page_num']}页" for item in pages)
            text = f"检索到{labels}"
        else:
            text = "这一轮没有检索到页面"
        return {
            "type": "retriever",
            "node": node,
            "text": text,
            "task": pages[0]["task"] if pages else "",
            "pages": pages,
        }
    if node == "grader":
        grade = str(output.get("grade") or "")
        return {"type": "grader", "node": node, "text": f"评估结果：{grade}", "grade": grade}
    if node == "rewrite":
        query = str(output.get("current_task") or "")
        return {
            "type": "rewrite",
            "node": node,
            "text": f"改写检索词：{query}",
            "query": query,
            "loop_count": output.get("loop_count"),
        }
    if node == "advance":
        task = str(output.get("current_task") or "")
        return {
            "type": "advance",
            "node": node,
            "text": f"进入下一子任务：{task}",
            "current_task": task,
            "task_index": output.get("task_index"),
        }
    if node == "generator":
        answer = str(output.get("final_answer") or "")
        return {"type": "answer", "node": node, "text": answer, "final_answer": answer}
    return None


def event_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    """只把六个节点自身的开始和结束推给前端，跳过模型调用内部事件。"""
    meta = event.get("metadata") or {}
    node = meta.get("langgraph_node")
    if node not in NODE_START_TEXT or event.get("name") != node:
        return None
    data = event.get("data") or {}
    kind = event.get("event")
    if kind == "on_chain_start":
        return start_event(node, data)
    if kind == "on_chain_end":
        output = data.get("output")
        if isinstance(output, dict):
            return end_event(node, output)
    return None


async def chat_events(message: str, thread_id: str):
    if graph is None:
        yield sse({"type": "error", "thread_id": thread_id, "text": "图还没有编译完成"})
        return
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    async with graph_lock:
        try:
            async for event in graph.astream_events(
                initial_state(message),
                config,
                version="v2",
            ):
                payload = event_payload(event)
                if payload is None:
                    continue
                payload["thread_id"] = thread_id
                yield sse(payload)
            yield sse({"type": "done", "thread_id": thread_id, "text": "本轮结束"})
        except Exception as exc:
            yield sse({"type": "error", "thread_id": thread_id, "text": str(exc)})


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global graph
    load_api_key()
    graph = await asyncio.to_thread(build_app)
    yield
    graph = None


app = FastAPI(title="ReflexRAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if graph is not None else "starting"}


@app.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    message = body.message.strip()
    thread_id = body.thread_id.strip() or "default"
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")
    if len(thread_id) > 128 or any(char in thread_id for char in "\r\n"):
        raise HTTPException(status_code=400, detail="thread_id 无效")
    return StreamingResponse(
        chat_events(message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

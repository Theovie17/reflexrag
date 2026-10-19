"""ReflexRAG 的 Streamlit 对话界面。

先启动 api_server.py，再运行 streamlit run app_ui.py。
所有请求使用同一个 thread_id，多轮对话走服务端的 MemorySaver。
"""

from __future__ import annotations

import json

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"
THREAD_ID = "reflexrag-ui"


def iter_sse(response: requests.Response):
    response.encoding = "utf-8"
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        yield json.loads(line[len("data:") :].strip())


def show_event(payload: dict, status) -> str:
    """把一条 SSE 写进思考面板，并返回要留在历史里的文字。"""
    kind = payload.get("type")
    text = str(payload.get("text") or "")
    if kind == "status":
        status.update(label=text or "正在处理")
        status.write(text)
        return text
    if kind == "planner":
        tasks = payload.get("sub_tasks") or []
        lines = [text] + [f"{index}. {task}" for index, task in enumerate(tasks, start=1)]
        for line in lines:
            status.write(line)
        return "\n".join(lines)
    if kind == "retriever":
        status.write(text)
        details = [text]
        for page in payload.get("pages") or []:
            preview = page.get("preview") or ""
            line = f"{page.get('company_name')}第{page.get('page_num')}页"
            if preview:
                line = f"{line}：{preview}"
            status.write(line)
            details.append(line)
        return "\n".join(details)
    if kind == "grader":
        line = f"评估结果：{payload.get('grade') or ''}"
        status.write(line)
        return line
    if kind in {"rewrite", "advance"}:
        status.write(text)
        return text
    if kind == "error":
        status.update(label="运行失败", state="error", expanded=True)
        status.write(text)
        return text
    return ""


def ask(question: str, status) -> tuple[str, list[str], str]:
    trace: list[str] = []
    answer = ""
    error = ""
    try:
        with requests.post(
            API_URL,
            json={"message": question, "thread_id": THREAD_ID},
            stream=True,
            timeout=(10, 300),
        ) as response:
            response.raise_for_status()
            for payload in iter_sse(response):
                line = show_event(payload, status)
                if line:
                    trace.append(line)
                if payload.get("type") == "answer":
                    answer = str(payload.get("final_answer") or payload.get("text") or "")
                elif payload.get("type") == "error":
                    error = str(payload.get("text") or "请求失败")
                elif payload.get("type") == "done" and not error:
                    status.update(label="思考完成", state="complete", expanded=False)
    except requests.RequestException as exc:
        error = f"无法连接 {API_URL}：{exc}"
        status.update(label="无法连接 API", state="error", expanded=True)
        status.write(error)
    return answer, trace, error


def draw_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        trace = message.get("trace") or []
        if trace:
            with st.expander("思考过程", expanded=False):
                st.markdown("\n\n".join(trace))
        if message.get("content"):
            st.markdown(message["content"])


def main() -> None:
    st.set_page_config(page_title="ReflexRAG", layout="centered")
    st.title("ReflexRAG")
    st.caption(f"多轮对话使用固定 thread_id：{THREAD_ID}。请先运行 api_server.py。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        draw_message(message)

    question = st.chat_input("请输入关于这两份债券报告的问题")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("正在连接 ReflexRAG...", expanded=True) as status:
            answer, trace, error = ask(question, status)
        if error:
            st.error(error)
            content = ""
        else:
            content = answer or "没有生成回答。"
            st.markdown(content)
        st.session_state.messages.append(
            {"role": "assistant", "content": content, "trace": trace, "error": error}
        )


if __name__ == "__main__":
    main()

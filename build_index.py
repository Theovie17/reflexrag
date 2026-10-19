"""把 children.json 写入本地 Chroma 向量库。

只向量化子块。父块留在 parents.json，按 parent_id 在内存里备用，不进入向量库。

依赖：langchain-huggingface、langchain-chroma、chromadb、sentence-transformers
模型：默认 BAAI/bge-small-zh-v1.5，缓存到 ./models。
需要更大的多语言模型时用 --model BAAI/bge-m3。
输出：./chroma_db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
FALLBACK_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "children"
TEST_QUERY = "中海地产的短期借款是多少？"

# BGE 检索要给查询加指令，文档侧不加。
QUERY_PROMPTS = {
    "BAAI/bge-m3": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
}
REQUIRED_METADATA = ("parent_id", "source_file", "page_num", "company_name")


def load_children(path: Path) -> list[Document]:
    records = json.loads(path.read_text(encoding="utf-8"))
    documents: list[Document] = []
    seen_ids: set[str] = set()
    for record in records:
        text = str(record.get("text") or "").strip()
        child_id = str(record.get("child_id") or "").strip()
        if not text or not child_id:
            continue
        if child_id in seen_ids:
            raise SystemExit(f"child_id 重复：{child_id}")
        seen_ids.add(child_id)

        raw_metadata = record.get("metadata") or {}
        missing = [key for key in REQUIRED_METADATA if raw_metadata.get(key) in (None, "")]
        if missing:
            raise SystemExit(f"{child_id} 缺少 metadata：{missing}")

        # Chroma 只接受字符串、数字、布尔值。页码保持整数，其余字段原样写入。
        metadata = {
            "parent_id": str(raw_metadata["parent_id"]),
            "source_file": str(raw_metadata["source_file"]),
            "page_num": int(raw_metadata["page_num"]),
            "company_name": str(raw_metadata["company_name"]),
            "child_id": child_id,
            "block_type": str(record.get("block_type") or ""),
        }
        documents.append(Document(page_content=text, metadata=metadata, id=child_id))
    if not documents:
        raise SystemExit(f"{path} 里没有可向量化的子块")
    return documents


def load_parents(path: Path) -> dict[str, dict]:
    """父块不向量化，只按 parent_id 放在内存里，供检索后取回整页上下文。"""
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["parent_id"]): record for record in records}


def model_is_cached(cache_folder: Path, model_name: str) -> bool:
    snapshot_root = cache_folder / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not snapshot_root.exists():
        return False
    for snapshot in snapshot_root.iterdir():
        if (snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists():
            return True
    return False


def build_embeddings(model_name: str, cache_folder: Path) -> HuggingFaceEmbeddings:
    cache_folder.mkdir(parents=True, exist_ok=True)
    query_prompt = QUERY_PROMPTS.get(model_name, "")
    query_encode_kwargs: dict = {"normalize_embeddings": True}
    if query_prompt:
        query_encode_kwargs["prompt"] = query_prompt
    # 权重已经在缓存里时不再访问网络。
    model_kwargs: dict = {"device": "cpu"}
    if model_is_cached(cache_folder, model_name):
        model_kwargs["local_files_only"] = True
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(cache_folder),
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        query_encode_kwargs=query_encode_kwargs,
        show_progress=True,
    )


def load_embeddings(model_name: str, cache_folder: Path) -> tuple[HuggingFaceEmbeddings, str]:
    try:
        print(f"加载嵌入模型 {model_name}")
        print(f"模型缓存目录 {cache_folder}")
        return build_embeddings(model_name, cache_folder), model_name
    except Exception as exc:
        if model_name == FALLBACK_MODEL:
            raise
        print(f"无法加载 {model_name}：{exc}", file=sys.stderr)
        print(f"改用已缓存的中文模型 {FALLBACK_MODEL}", file=sys.stderr)
        return build_embeddings(FALLBACK_MODEL, cache_folder), FALLBACK_MODEL


def build_vectorstore(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Path,
    model_name: str,
) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return Chroma.from_documents(
        documents=documents,
        ids=[doc.metadata["child_id"] for doc in documents],
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
        },
        client=client,
    )


def preview(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "…"


def run_test_query(
    vectorstore: Chroma,
    parents: dict[str, dict],
    query: str,
    top_k: int,
) -> None:
    hits = vectorstore.similarity_search_with_score(query, k=top_k)
    print(f"\n查询：{query}")
    print(f"Top {len(hits)}")
    if not hits:
        print("没有检索到子块")
        return
    for rank, (doc, distance) in enumerate(hits, start=1):
        metadata = doc.metadata
        parent_id = metadata.get("parent_id", "")
        parent = parents.get(str(parent_id))
        print(f"\n[{rank}] distance={distance:.4f}")
        print(f"parent_id: {parent_id}")
        print(f"child_id: {metadata.get('child_id', '')}")
        print(f"company_name: {metadata.get('company_name', '')}")
        print(f"page_num: {metadata.get('page_num', '')}")
        print(f"source_file: {metadata.get('source_file', '')}")
        if parent:
            print(f"父块备用: 已在内存中，{parent.get('char_count', len(parent.get('text', '')))} 字")
        print(preview(doc.page_content))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="用子块构建 Chroma 向量库并做一次检索")
    parser.add_argument("--children", type=Path, default=script_dir / "children.json")
    parser.add_argument("--parents", type=Path, default=script_dir / "parents.json")
    parser.add_argument("--persist-dir", type=Path, default=script_dir / "chroma_db")
    parser.add_argument("--cache-dir", type=Path, default=script_dir / "models")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace 嵌入模型")
    parser.add_argument("--query", default=TEST_QUERY)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    documents = load_children(args.children)
    parents = load_parents(args.parents)
    print(f"子块 {len(documents)} 条，准备向量化")
    print(f"父块 {len(parents)} 条，只留在内存里，不写入向量库")

    embeddings, model_name = load_embeddings(args.model, args.cache_dir)
    vectorstore = build_vectorstore(documents, embeddings, args.persist_dir, model_name)
    stored = vectorstore._collection.count()
    print(f"已写入 {args.persist_dir}，集合 {COLLECTION_NAME}，共 {stored} 条")
    print(f"嵌入模型 {model_name}")
    if stored != len(documents):
        raise SystemExit(f"写入条数 {stored} 与子块数 {len(documents)} 不一致")

    if not args.skip_test:
        run_test_query(vectorstore, parents, args.query, args.top_k)


if __name__ == "__main__":
    main()

"""把债券半年度报告 PDF 解析为父子文本块。

策略：
- 用 pdfplumber 识别表格，转成 Markdown。父块保留整表，不按字数把表切开。
- 父块以页面为单位。一页内容超过约 1500 字时，只在段落边界再切开，parent_id
  形如「中海地产_page_12」；同一页的后续片段为「中海地产_page_12_2」。
- 纯文本子块按自然段落和句末标点打包到 200–300 字。
- 表格按行切成子块：第一行是表头，每一数据行写成
  [表头1: 值1, 表头2: 值2]，并挂在同一个 parent_id 下。
- 子块 metadata 含 parent_id、source_file、page_num、company_name。

依赖：pdfplumber
输出：parents.json、children.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pdfplumber


PARENT_TARGET = 1500
CHILD_MIN = 200
CHILD_MAX = 300
# 页尾不足一块时，允许并回上一块，避免留下几十个字的碎片。
CHILD_TAIL_MAX = 300

# Word 导出的表格会把单元格内边距画成约 13.6pt 的短线。这些短线若在合并前
# 被保留，会被接成假列。先按 16pt 预过滤，再识别表格。
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "edge_min_length_prefilter": 16,
    "edge_min_length": 20,
    "snap_x_tolerance": 3,
    "snap_y_tolerance": 3,
    "join_x_tolerance": 3,
    "join_y_tolerance": 3,
    "intersection_tolerance": 5,
}

COMPANY_RULES = (
    ("中海", "中海地产"),
    ("四川发展", "四川发展"),
    ("川发", "四川发展"),
)

SENTENCE_END = "。！？；"
MAJOR_HEADING_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+节")
HEADING_RE = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百零〇0-9]+节"
    r"|[一二三四五六七八九十]+、"
    r"|（[一二三四五六七八九十]+）"
    r"|\d{1,2}[.．、]"
    r"|目录|重要提示|重大风险提示|释义|备查文件"
    r")"
)
TOC_ENTRY_RE = re.compile(r"(?:[.·…]{4,}|—{4,})\s*\d+\s*$")
# 目录点引导符在 PDF 里经常被抽成一长串普通句点。
TOC_DOTS_RE = re.compile(r"\.{6,}\s*\d+\s*$")

PUNCT_SPLIT_RE = re.compile(r"(?<=[。！？；])")
COMMA_SPLIT_RE = re.compile(r"(?<=[，、,])")
AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2}")
CAPTION_RE = re.compile(
    r"^(?:单位：[0-9A-Za-z\u4e00-\u9fff%．.]{1,16})?(?:币种：[0-9A-Za-z\u4e00-\u9fff]{1,16})?$"
)
HEADER_HINTS = ("项目", "年度", "本期", "期末", "期初", "比例", "类别", "名称", "日期", "构成")


@dataclass
class Block:
    kind: str  # "text" or "table"
    text: str
    top: float
    grid: list[list[str]] | None = None


@dataclass
class ParentChunk:
    parent_id: str
    text: str
    char_count: int
    metadata: dict
    blocks: list[Block] = field(default_factory=list, repr=False)

    def metadata_for_child(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "source_file": self.metadata["source_file"],
            "page_num": self.metadata["page_num"],
            "company_name": self.metadata["company_name"],
        }


@dataclass
class TableFlowState:
    """跨页表格的表头，以及上一张表，用来接上被页面切开的半个字。"""

    last_table: Block | None = None
    headers: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class ChildChunk:
    child_id: str
    text: str
    block_type: str
    metadata: dict


def company_name_from_filename(filename: str) -> str:
    for keyword, name in COMPANY_RULES:
        if keyword in filename:
            return name
    raise ValueError(f"无法从文件名识别公司：{filename}")


def clean_inline(text: str) -> str:
    """去掉换行和私用区符号，保留单元格里的完整文字。"""
    if not text:
        return ""
    text = text.replace("\uf052", "☑").replace("\uf0fe", "☑").replace("\uf0a3", "☐")
    text = re.sub(r"[\uE000-\uF8FF]", "", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("\r", "\n")
    parts = []
    for piece in text.split("\n"):
        piece = piece.strip()
        if not piece:
            continue
        if not parts:
            parts.append(piece)
            continue
        prev = parts[-1]
        if prev[-1].isascii() and prev[-1].isalnum() and piece[0].isascii() and piece[0].isalnum():
            parts[-1] = prev + " " + piece
        else:
            parts[-1] = prev + piece
    text = parts[0] if parts else ""
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_heading(text: str) -> bool:
    return bool(HEADING_RE.match(text.strip()))


def is_major_heading(text: str) -> bool:
    return bool(MAJOR_HEADING_RE.match(text.strip()))


def looks_like_toc_entry(text: str) -> bool:
    stripped = text.strip()
    return bool(TOC_ENTRY_RE.search(stripped) or TOC_DOTS_RE.search(stripped))


def smart_concat(left: str, right: str) -> str:
    """中文折行直接拼接；英文或数字在折行处补一个空格。"""
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return left + " " + right
    return left + right


def union_length(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    total += end - start
    return total


def drop_contained_tables(tables: list, tolerance: float = 2.0) -> list:
    """去掉落在更大表格内部的假表（表头单元格常被单独识别成表）。"""
    kept = []
    for index, table in enumerate(tables):
        x0, top, x1, bottom = table.bbox
        area = max(0.0, x1 - x0) * max(0.0, bottom - top)
        contained = False
        for other_index, other in enumerate(tables):
            if index == other_index:
                continue
            ox0, otop, ox1, obottom = other.bbox
            other_area = max(0.0, ox1 - ox0) * max(0.0, obottom - otop)
            if other_area <= area + 1:
                continue
            if (
                x0 >= ox0 - tolerance
                and top >= otop - tolerance
                and x1 <= ox1 + tolerance
                and bottom <= obottom + tolerance
            ):
                contained = True
                break
        if not contained:
            kept.append(table)
    return kept


def drop_empty_lines(grid: list[list[str]]) -> list[list[str]]:
    if not grid:
        return []
    width = max(len(row) for row in grid)
    normalized = [row + [""] * (width - len(row)) for row in grid]
    nonempty_cols = [
        col
        for col in range(width)
        if any(normalized[row][col].strip() for row in range(len(normalized)))
    ]
    trimmed = [[row[col].strip() for col in nonempty_cols] for row in normalized]
    return [row for row in trimmed if any(cell.strip() for cell in row)]


def grid_from_extract(table) -> list[list[str]]:
    raw = table.extract() or []
    grid = []
    for row in raw:
        grid.append([clean_inline("" if cell is None else str(cell)) for cell in row])
    return drop_empty_lines(grid)


def strong_rule_positions(
    page,
    bbox: tuple[float, float, float, float],
    orientation: str,
    min_fraction: float = 0.5,
    cluster_tolerance: float = 2.5,
) -> list[float]:
    """保留贯穿表格大部分高度或宽度的线，丢掉表头里的短装饰线。"""
    x0, top, x1, bottom = bbox
    span = (bottom - top) if orientation == "v" else (x1 - x0)
    if span <= 0:
        return []

    raw: list[tuple[float, float, float]] = []
    for edge in page.edges:
        if edge.get("orientation") != orientation:
            continue
        if orientation == "v":
            position = edge["x0"]
            if not (x0 - 3 <= position <= x1 + 3):
                continue
            start = max(min(edge["top"], edge["bottom"]), top)
            end = min(max(edge["top"], edge["bottom"]), bottom)
        else:
            position = edge["top"]
            if not (top - 3 <= position <= bottom + 3):
                continue
            start = max(min(edge["x0"], edge["x1"]), x0)
            end = min(max(edge["x0"], edge["x1"]), x1)
        if end - start >= 8:
            raw.append((position, start, end))

    raw.sort(key=lambda item: item[0])
    clusters: list[list] = []
    for position, start, end in raw:
        if clusters and abs(position - clusters[-1][0]) <= cluster_tolerance:
            count = clusters[-1][2]
            clusters[-1][0] = (clusters[-1][0] * count + position) / (count + 1)
            clusters[-1][1].append((start, end))
            clusters[-1][2] = count + 1
        else:
            clusters.append([position, [(start, end)], 1])

    kept = [
        position
        for position, intervals, _count in clusters
        if union_length(intervals) >= min_fraction * span
    ]
    kept.sort()
    deduped: list[float] = []
    for position in kept:
        if not deduped or position - deduped[-1] > cluster_tolerance:
            deduped.append(position)
    return deduped


def grid_from_words(page, bbox: tuple[float, float, float, float]) -> list[list[str]]:
    """按贯穿全表的横竖线把词填回单元格，避免合并单元格造成的列错位。"""
    xs = strong_rule_positions(page, bbox, "v")
    ys = strong_rule_positions(page, bbox, "h")
    if len(xs) < 3 or len(ys) < 3:
        return []

    x0, top, x1, bottom = bbox
    words = []
    for word in page.extract_words():
        center_x = (word["x0"] + word["x1"]) / 2
        center_y = (word["top"] + word["bottom"]) / 2
        if x0 - 1 <= center_x <= x1 + 1 and top - 1 <= center_y <= bottom + 1:
            words.append(word)

    cols = len(xs) - 1
    rows = len(ys) - 1
    cells: list[list[list]] = [[[] for _ in range(cols)] for _ in range(rows)]
    for word in words:
        center_x = (word["x0"] + word["x1"]) / 2
        center_y = (word["top"] + word["bottom"]) / 2
        col = next((i for i in range(cols) if xs[i] <= center_x < xs[i + 1]), None)
        row = next((i for i in range(rows) if ys[i] <= center_y < ys[i + 1]), None)
        if col is None and center_x >= xs[-1]:
            col = cols - 1
        if row is None and center_y >= ys[-1]:
            row = rows - 1
        if col is None or row is None:
            continue
        cells[row][col].append(word)

    grid: list[list[str]] = []
    for row in cells:
        text_row = []
        for bucket in row:
            text_row.append(clean_inline(join_words(bucket)))
        grid.append(text_row)
    return drop_empty_lines(grid)


def join_words(words: list[dict]) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda word: (round(word["top"], 1), word["x0"]))
    lines: list[list[dict]] = []
    for word in ordered:
        if not lines or abs(word["top"] - lines[-1][0]["top"]) > 2.5:
            lines.append([word])
        else:
            lines[-1].append(word)
    pieces = []
    for line in lines:
        line = sorted(line, key=lambda word: word["x0"])
        text = ""
        prev_x1 = None
        for word in line:
            token = word["text"]
            if prev_x1 is not None and word["x0"] - prev_x1 > 8:
                text += " "
            text = smart_concat(text, token) if text else token
            prev_x1 = word["x1"]
        pieces.append(text)
    return clean_inline("\n".join(pieces))


def grid_score(grid: list[list[str]]) -> tuple[float, int]:
    """空单元格比例越低越好；同样稀疏时保留文字更多的网格。"""
    cells = [cell.strip() for row in grid for cell in row]
    if not cells:
        return (1.0, 0)
    nonempty_chars = sum(len(cell) for cell in cells)
    empty_ratio = sum(1 for cell in cells if not cell) / len(cells)
    return (empty_ratio, -nonempty_chars)


def choose_grid(page, table) -> list[list[str]]:
    extracted = grid_from_extract(table)
    rebuilt = grid_from_words(page, table.bbox)
    candidates = [
        grid for grid in (extracted, rebuilt) if grid and len(grid) >= 1 and len(grid[0]) >= 2
    ]
    if not candidates:
        return []
    return min(candidates, key=grid_score)


def grid_to_markdown(grid: list[list[str]]) -> str:
    if not grid or len(grid[0]) < 2:
        return ""
    width = len(grid[0])

    def escape(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", "")

    header = [escape(cell) for cell in grid[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in grid[1:]:
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(escape(cell) for cell in padded[:width]) + " |")
    return "\n".join(lines)


def word_in_any_bbox(word: dict, bboxes: list[tuple[float, float, float, float]]) -> bool:
    center_x = (word["x0"] + word["x1"]) / 2
    center_y = (word["top"] + word["bottom"]) / 2
    for x0, top, x1, bottom in bboxes:
        if x0 - 1 <= center_x <= x1 + 1 and top - 1 <= center_y <= bottom + 1:
            return True
    return False


def is_running_header(text: str, top: float) -> bool:
    return top < 70 and "公司债券" in text and "报告" in text


def is_page_number(text: str, top: float, page_height: float) -> bool:
    return top > page_height - 75 and bool(re.fullmatch(r"\d{1,3}", text.strip()))


def lines_from_words(words: list[dict]) -> list[dict]:
    ordered = sorted(words, key=lambda word: (word["top"], word["x0"]))
    raw_lines: list[list[dict]] = []
    for word in ordered:
        if not raw_lines or abs(word["top"] - raw_lines[-1][0]["top"]) > 3:
            raw_lines.append([word])
        else:
            raw_lines[-1].append(word)

    lines = []
    for raw in raw_lines:
        raw = sorted(raw, key=lambda word: word["x0"])
        text = ""
        prev_x1 = None
        for word in raw:
            token = clean_inline(word["text"])
            if not token:
                continue
            if prev_x1 is not None and word["x0"] - prev_x1 > 8:
                text += " "
            text = smart_concat(text, token) if text else token
            prev_x1 = word["x1"]
        text = text.strip()
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "x0": raw[0]["x0"],
                "x1": max(word["x1"] for word in raw),
                "top": min(word["top"] for word in raw),
                "bottom": max(word["bottom"] for word in raw),
            }
        )
    return lines


def paragraphs_from_lines(lines: list[dict]) -> list[dict]:
    """把视觉行收成自然段。满行且未到句末的，视为折行而不是新段落。"""
    paragraphs: list[dict] = []
    buffer = ""
    buffer_top = 0.0
    prev = None

    def flush() -> None:
        nonlocal buffer, buffer_top
        text = buffer.strip()
        if text:
            paragraphs.append({"text": text, "top": buffer_top})
        buffer = ""

    for line in lines:
        text = line["text"]
        if prev is None:
            buffer = text
            buffer_top = line["top"]
            prev = line
            continue

        gap = line["top"] - prev["bottom"]
        prev_text = prev["text"]
        start_new = False
        if gap > 22:
            start_new = True
        elif is_heading(text) or is_heading(prev_text):
            start_new = True
        elif looks_like_toc_entry(prev_text):
            start_new = True
        elif re.search(rf"[{SENTENCE_END}：:]$", prev_text.strip()):
            start_new = True
        elif prev["x1"] < 480:
            start_new = True

        if start_new:
            flush()
            buffer = text
            buffer_top = line["top"]
        else:
            buffer = smart_concat(buffer, text)
        prev = line

    flush()
    return paragraphs


def is_caption(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    if "单位：" not in compact and "币种：" not in compact:
        return False
    return bool(CAPTION_RE.fullmatch(compact))


def row_has_amount(row: list[str]) -> bool:
    return any(AMOUNT_RE.search(cell) for cell in row)


def looks_like_header(row: list[str]) -> bool:
    if len(row) < 3 or row_has_amount(row):
        return False
    joined = "".join(row)
    if not any(hint in joined for hint in HEADER_HINTS):
        return False
    return sum(1 for cell in row if cell.strip()) >= 2


# 这些字出现在单元格末尾时，标签通常还没写完，下一页会接上「金」「元）」这类尾巴。
INCOMPLETE_LABEL_END = set("和及与或（(亿现")


def orphan_fragment(row: list[str]) -> str:
    """页面顶上只剩「金」或「元）」时，是上一页单元格被切开的尾巴。"""
    nonempty = [(index, cell.strip()) for index, cell in enumerate(row) if cell.strip()]
    if len(nonempty) != 1 or nonempty[0][0] != 0:
        return ""
    text = nonempty[0][1]
    if re.fullmatch(r"[\u4e00-\u9fff]{1,2}", text):
        return text
    if re.fullmatch(r"[\u4e00-\u9fff]{1,3}）", text):
        return text
    return ""


def can_absorb_fragment(previous: list[list[str]]) -> bool:
    """上一页标签被截断，且取值还留在上一行时，才把下一页的短尾巴接回去。"""
    if not previous:
        return False
    label = previous[-1][0].strip()
    if len(label) < 8 or re.search(rf"[{SENTENCE_END}：:]$", label):
        return False
    if any(cell.strip() for cell in previous[-1][1:]):
        return True
    return label[-1] in INCOMPLETE_LABEL_END


def repair_table_flow(blocks: list[Block], state: TableFlowState) -> list[Block]:
    """接上跨页断开的单元格，并为续页表格补上上一张表的表头。"""
    kept: list[Block] = []
    for block in blocks:
        if block.kind == "text":
            fragment = block.text.strip()
            if (
                state.last_table
                and state.last_table.grid
                and "\n" not in fragment
                and block.top < 130
                and len(fragment) <= 8
                and not is_heading(fragment)
                and not is_caption(fragment)
                and can_absorb_fragment(state.last_table.grid)
            ):
                state.last_table.grid[-1][0] += fragment
                state.last_table.text = grid_to_markdown(state.last_table.grid)
                continue
            kept.append(block)
            continue
        if block.kind != "table" or not block.grid:
            kept.append(block)
            continue
        grid = block.grid
        fragment = orphan_fragment(grid[0]) if grid else ""
        if (
            fragment
            and state.last_table
            and state.last_table.grid
            and can_absorb_fragment(state.last_table.grid)
        ):
            state.last_table.grid[-1][0] += fragment
            state.last_table.text = grid_to_markdown(state.last_table.grid)
            grid = grid[1:]
        if not grid:
            continue
        width = len(grid[0])
        header = state.headers.get(width)
        # 只给带金额的报表续页补表头，避免把资产负债表表头套到键值表上。
        if (
            header
            and width >= 3
            and not looks_like_header(grid[0])
            and any(row_has_amount(row) for row in grid[:6])
        ):
            grid = [list(header)] + grid
        block.grid = grid
        block.text = grid_to_markdown(grid)
        if not block.text:
            continue
        if looks_like_header(grid[0]):
            state.headers[width] = list(grid[0])
        state.last_table = block
        kept.append(block)
    return kept


def extract_page_blocks(page) -> list[Block]:
    tables = drop_contained_tables(page.find_tables(TABLE_SETTINGS) or [])
    table_blocks: list[Block] = []
    bboxes = []
    for table in tables:
        grid = choose_grid(page, table)
        markdown = grid_to_markdown(grid)
        if not markdown:
            continue
        table_blocks.append(Block("table", markdown, table.bbox[1], grid))
        bboxes.append(table.bbox)

    words = []
    for word in page.extract_words() or []:
        if word_in_any_bbox(word, bboxes):
            continue
        words.append(word)

    lines = lines_from_words(words)
    lines = [
        line
        for line in lines
        if not is_running_header(line["text"], line["top"])
        and not is_page_number(line["text"], line["top"], page.height)
    ]
    text_blocks = [
        Block("text", paragraph["text"], paragraph["top"])
        for paragraph in paragraphs_from_lines(lines)
        if paragraph["text"].strip()
    ]
    blocks = text_blocks + table_blocks
    blocks.sort(key=lambda block: (block.top, 0 if block.kind == "text" else 1))
    return blocks


def group_parent_blocks(blocks: list[Block]) -> list[list[Block]]:
    """一页默认一个父块。超过约 1500 字时在段落或大章节标题处切开，表格不切开。"""
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            groups.append(current)
        current = []
        current_len = 0

    for block in blocks:
        block_len = len(block.text)
        if current and block.kind == "text" and is_major_heading(block.text) and current_len >= 700:
            flush()
        elif (
            current
            and block.kind == "text"
            and current_len >= 800
            and current_len + block_len > PARENT_TARGET
        ):
            flush()
        elif current and block.kind == "table" and current_len >= PARENT_TARGET:
            flush()

        current.append(block)
        current_len += block_len
        if current_len >= PARENT_TARGET and not (
            block.kind == "table" and current_len == block_len and len(current) > 1
        ):
            # 表格自己超长时留在当前父块里，和它前面的短标题在一起，然后立刻收口。
            if block.kind == "table" or current_len >= PARENT_TARGET:
                flush()

    flush()
    return groups


def parent_id_for(company: str, page_num: int, part_index: int, part_count: int) -> str:
    if part_count == 1 or part_index == 1:
        return f"{company}_page_{page_num}"
    return f"{company}_page_{page_num}_{part_index}"


def render_parent_text(blocks: list[Block]) -> str:
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())


def split_oversized_paragraph(text: str) -> list[str]:
    sentences = [part for part in PUNCT_SPLIT_RE.split(text) if part.strip()]
    if not sentences:
        sentences = [text]
    pieces: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) <= CHILD_MAX:
            pieces.append(sentence)
            continue
        clauses = [part for part in COMMA_SPLIT_RE.split(sentence) if part.strip()]
        clause_buf = ""
        for clause in clauses:
            clause = clause.strip()
            if len(clause) > CHILD_MAX:
                if clause_buf:
                    pieces.append(clause_buf)
                    clause_buf = ""
                for start in range(0, len(clause), CHILD_MAX):
                    pieces.append(clause[start : start + CHILD_MAX])
                continue
            if clause_buf and len(clause_buf) + len(clause) > CHILD_MAX:
                pieces.append(clause_buf)
                clause_buf = clause
            else:
                clause_buf = smart_concat(clause_buf, clause)
        if clause_buf:
            pieces.append(clause_buf)
    return [piece.strip() for piece in pieces if piece.strip()]


def pack_text_paragraphs(paragraphs: list[str]) -> list[tuple[str, int]]:
    """短段合并到 200–300 字。返回 (文本, 起始段落序号)。"""
    units: list[tuple[str, int]] = []
    for index, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > CHILD_MAX:
            units.extend((piece, index) for piece in split_oversized_paragraph(paragraph))
        else:
            units.append((paragraph, index))

    chunks: list[tuple[str, int]] = []
    buffer = ""
    buffer_index = 0
    for text, index in units:
        if not buffer:
            buffer = text
            buffer_index = index
            continue
        combined = buffer + "\n" + text
        if len(combined) <= CHILD_MAX:
            buffer = combined
            continue
        chunks.append((buffer, buffer_index))
        buffer = text
        buffer_index = index
    if buffer:
        if (
            chunks
            and len(buffer) < CHILD_MIN
            and len(chunks[-1][0]) + 1 + len(buffer) <= CHILD_TAIL_MAX
        ):
            chunks[-1] = (chunks[-1][0] + "\n" + buffer, chunks[-1][1])
        else:
            chunks.append((buffer, buffer_index))
    return [(text.strip(), index) for text, index in chunks if text.strip()]


def format_header_row(header: list[str], row: list[str]) -> str:
    """把一行写成 [表头1: 值1, 表头2: 值2]。空值不写入，避免整表空白列稀释特征。"""
    pairs: list[str] = []
    for index, name in enumerate(header):
        value = row[index].strip() if index < len(row) and row[index] else ""
        name = name.strip()
        if not name or not value:
            continue
        pairs.append(f"{name}: {value}")
    if not pairs:
        return ""
    return "[" + ", ".join(pairs) + "]"


def table_row_texts(grid: list[list[str]]) -> list[str]:
    """第一行作表头，其余每一行一个子块。两列键值表没有列表头，每行就是一条字段。"""
    if not grid or not grid[0]:
        return []
    if len(grid[0]) == 2 and not looks_like_header(grid[0]):
        texts: list[str] = []
        for row in grid:
            left = row[0].strip() if row and row[0] else ""
            right = row[1].strip() if len(row) > 1 and row[1] else ""
            if left and right:
                texts.append(f"[{left}: {right}]")
            elif left or right:
                texts.append(f"[{left or right}]")
        return texts

    header = [cell.strip() for cell in grid[0]]
    texts = [format_header_row(header, row) for row in grid[1:]]
    texts = [text for text in texts if text]
    if texts:
        return texts
    if len(header) == 2 and header[0] and header[1]:
        return [f"[{header[0]}: {header[1]}]"]
    return []


def build_children(parent: ParentChunk) -> list[ChildChunk]:
    """纯文本按 200–300 字切。表格按行切，所有行子块共用这个父块的 parent_id。"""
    paragraphs: list[str] = []
    pending_heading = ""
    pending_caption = ""
    table_prefix: dict[int, str] = {}

    for index, block in enumerate(parent.blocks):
        if block.kind == "text":
            if is_caption(block.text):
                pending_caption = (pending_caption + " " + block.text.strip()).strip()
                continue
            paragraphs.append(block.text.strip())
            pending_heading = block.text.strip() if len(block.text.strip()) <= 40 else ""
            continue
        parts = [part for part in (pending_heading, pending_caption) if part and part not in block.text]
        if parts:
            table_prefix[index] = " ".join(parts)
        pending_heading = ""
        pending_caption = ""

    packed = pack_text_paragraphs(paragraphs)
    chunks_at: dict[int, list[str]] = {}
    for text, source_index in packed:
        chunks_at.setdefault(source_index, []).append(text)

    children: list[ChildChunk] = []
    emitted: set[int] = set()
    paragraph_index = 0
    meta = parent.metadata_for_child()
    company = str(parent.metadata.get("company_name") or "")
    for index, block in enumerate(parent.blocks):
        if block.kind == "text":
            if is_caption(block.text):
                continue
            if paragraph_index not in emitted and paragraph_index in chunks_at:
                for text in chunks_at[paragraph_index]:
                    children.append(
                        ChildChunk(child_id="", text=text, block_type="text", metadata=dict(meta))
                    )
                emitted.add(paragraph_index)
            paragraph_index += 1
            continue
        context = " ".join(part for part in (company, table_prefix.get(index, "")) if part).strip()
        row_texts = table_row_texts(block.grid or [])
        if not row_texts:
            fallback = block.text.strip()
            if context and context not in fallback:
                fallback = context + "\n" + fallback
            if fallback:
                children.append(
                    ChildChunk(child_id="", text=fallback, block_type="table_row", metadata=dict(meta))
                )
            continue
        for row_text in row_texts:
            text = f"{context} {row_text}".strip() if context else row_text
            children.append(
                ChildChunk(child_id="", text=text, block_type="table_row", metadata=dict(meta))
            )

    for index, child in enumerate(children, start=1):
        child.child_id = f"{parent.parent_id}_c{index}"
        child.metadata["parent_id"] = parent.parent_id
    return children


def process_pdf(path: Path) -> tuple[list[ParentChunk], list[ChildChunk]]:
    company = company_name_from_filename(path.name)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    page_blocks: list[tuple[int, list[Block]]] = []
    flow = TableFlowState()
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            try:
                blocks = repair_table_flow(extract_page_blocks(page), flow)
            except Exception as exc:  # 单页失败不丢掉整份报告
                print(f"警告：{path.name} 第 {page_num} 页解析失败：{exc}", file=sys.stderr)
                continue
            if blocks:
                page_blocks.append((page_num, blocks))

    for page_num, blocks in page_blocks:
        groups = group_parent_blocks(blocks)
        for part_index, group in enumerate(groups, start=1):
            parent_id = parent_id_for(company, page_num, part_index, len(groups))
            text = render_parent_text(group)
            parent = ParentChunk(
                parent_id=parent_id,
                text=text,
                char_count=len(text),
                metadata={
                    "source_file": path.name,
                    "page_num": page_num,
                    "company_name": company,
                    "part": part_index,
                },
                blocks=group,
            )
            parents.append(parent)
            children.extend(build_children(parent))
    return parents, children


def chunk_to_dict(chunk: ParentChunk | ChildChunk) -> dict:
    if isinstance(chunk, ParentChunk):
        return {
            "parent_id": chunk.parent_id,
            "text": chunk.text,
            "char_count": chunk.char_count,
            "metadata": chunk.metadata,
        }
    return {
        "child_id": chunk.child_id,
        "text": chunk.text,
        "block_type": chunk.block_type,
        "metadata": chunk.metadata,
    }


def length_histogram(lengths: list[int]) -> dict[str, int]:
    buckets = {"<200": 0, "200-300": 0, "301-500": 0, "501-1500": 0, "1501-2500": 0, ">2500": 0}
    for length in lengths:
        if length < 200:
            buckets["<200"] += 1
        elif length <= 300:
            buckets["200-300"] += 1
        elif length <= 500:
            buckets["301-500"] += 1
        elif length <= 1500:
            buckets["501-1500"] += 1
        elif length <= 2500:
            buckets["1501-2500"] += 1
        else:
            buckets[">2500"] += 1
    return buckets


def validate(parents: list[ParentChunk], children: list[ChildChunk]) -> None:
    parent_ids = {parent.parent_id for parent in parents}
    if len(parent_ids) != len(parents):
        raise SystemExit("parent_id 有重复")
    required = ("parent_id", "source_file", "page_num", "company_name")
    for child in children:
        missing = [key for key in required if child.metadata.get(key) in (None, "")]
        if missing:
            raise SystemExit(f"{child.child_id} 缺少 metadata：{missing}")
        if child.metadata["parent_id"] not in parent_ids:
            raise SystemExit(f"{child.child_id} 的 parent_id 不存在")
        if not child.text.strip():
            raise SystemExit(f"{child.child_id} 文本为空")


def print_summary(parents: list[ParentChunk], children: list[ChildChunk]) -> None:
    print(f"父块 {len(parents)}  子块 {len(children)}")
    by_company: dict[str, dict[str, int]] = {}
    for parent in parents:
        name = parent.metadata["company_name"]
        bucket = by_company.setdefault(name, {"parents": 0, "children": 0, "tables": 0})
        bucket["parents"] += 1
    for child in children:
        name = child.metadata["company_name"]
        bucket = by_company.setdefault(name, {"parents": 0, "children": 0, "tables": 0})
        bucket["children"] += 1
        if child.block_type == "table_row":
            bucket["tables"] += 1
    for name, bucket in by_company.items():
        print(
            f"  {name}: 父块 {bucket['parents']}  子块 {bucket['children']}"
            f"  其中表格子块 {bucket['tables']}"
        )
    print("父块字数分布:", length_histogram([parent.char_count for parent in parents]))
    text_lengths = [len(child.text) for child in children if child.block_type == "text"]
    table_lengths = [len(child.text) for child in children if child.block_type == "table_row"]
    print("文本子块字数分布:", length_histogram(text_lengths))
    print("表格行子块字数分布:", length_histogram(table_lengths))
    if text_lengths:
        print(
            f"文本子块字数: min={min(text_lengths)} median={sorted(text_lengths)[len(text_lengths)//2]}"
            f" max={max(text_lengths)}"
        )


def discover_pdfs(input_dir: Path) -> list[Path]:
    pdfs = sorted(path for path in input_dir.glob("*.pdf") if path.is_file())
    if not pdfs:
        raise SystemExit(f"目录里没有 PDF：{input_dir}")
    return pdfs


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="将财报 PDF 切成父子块并写入 JSON")
    parser.add_argument("--input-dir", type=Path, default=script_dir, help="PDF 所在目录")
    parser.add_argument("--output-dir", type=Path, default=script_dir, help="JSON 输出目录")
    args = parser.parse_args()

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    for path in discover_pdfs(args.input_dir):
        print(f"解析 {path.name}")
        file_parents, file_children = process_pdf(path)
        parents.extend(file_parents)
        children.extend(file_children)

    validate(parents, children)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parents_path = args.output_dir / "parents.json"
    children_path = args.output_dir / "children.json"
    parents_path.write_text(
        json.dumps([chunk_to_dict(chunk) for chunk in parents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    children_path.write_text(
        json.dumps([chunk_to_dict(chunk) for chunk in children], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(parents, children)
    print(f"已写入 {parents_path}")
    print(f"已写入 {children_path}")


if __name__ == "__main__":
    main()

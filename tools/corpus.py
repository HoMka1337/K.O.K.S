import bz2
import gzip
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Dict, Optional

from config import Config


MIN_PARAGRAPH_CHARS = 80
MAX_PARAGRAPH_CHARS = 2000
SKIP_PREFIXES = (
    "== См. также ==", "== Примечания ==", "== Литература ==",
    "== Ссылки ==", "Категория:", "Файл:", "Шаблон:",
)


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\[http[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_wiki_xml(path: Path) -> Iterator[Dict[str, str]]:
    with bz2.open(path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag.endswith("page"):
                title_el = elem.find(".//{*}title")
                text_el = elem.find(".//{*}revision/{*}text")
                if title_el is not None and text_el is not None:
                    title = title_el.text or ""
                    body = text_el.text or ""
                    if body and not title.startswith(("Википедия:", "Шаблон:", "Категория:")):
                        yield {"title": title, "text": body}
                elem.clear()


def iter_wiki_lines(path: Path) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        title = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if title is None:
                title = line
            else:
                yield {"title": title, "text": line}
                title = None


def iter_text(path: Path) -> Iterator[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            buf = []
            for line in f:
                line = line.rstrip()
                if not line and buf:
                    yield {"title": path.stem, "text": " ".join(buf)}
                    buf = []
                else:
                    buf.append(line)
            if buf:
                yield {"title": path.stem, "text": " ".join(buf)}
    except OSError:
        return


def iter_gzip(path: Path) -> Iterator[Dict[str, str]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            buf = []
            for line in f:
                line = line.rstrip()
                if not line and buf:
                    yield {"title": path.stem, "text": " ".join(buf)}
                    buf = []
                else:
                    buf.append(line)
            if buf:
                yield {"title": path.stem, "text": " ".join(buf)}
    except OSError:
        return


def iter_jsonl(path: Path) -> Iterator[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = obj.get("title") or obj.get("id") or path.stem
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    yield {"title": str(title), "text": str(text)}
    except OSError:
        return


def iter_corpus(path: Path) -> Iterator[Dict[str, str]]:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".bz2" or name.endswith(".xml.bz2"):
        yield from iter_wiki_xml(path)
    elif suffix == ".gz":
        yield from iter_gzip(path)
    elif suffix == ".jsonl" or suffix == ".ndjson":
        yield from iter_jsonl(path)
    elif suffix in (".xml", ".txt", ".md"):
        if suffix == ".xml":
            yield from iter_wiki_lines(path)
        else:
            yield from iter_text(path)
    else:
        yield from iter_text(path)


def paragraphs_from_text(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    buf = []
    size = 0
    for s in parts:
        buf.append(s)
        size += len(s)
        if size >= MIN_PARAGRAPH_CHARS:
            chunk = " ".join(buf).strip()
            if MIN_PARAGRAPH_CHARS <= len(chunk) <= MAX_PARAGRAPH_CHARS:
                out.append(chunk)
            buf = []
            size = 0
    if buf:
        chunk = " ".join(buf).strip()
        if MIN_PARAGRAPH_CHARS <= len(chunk) <= MAX_PARAGRAPH_CHARS:
            out.append(chunk)
    return out


def build_corpus_jsonl(input_paths: List[Path], output_path: Path,
                       max_docs: int = 0, max_chars: int = 0) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    total_chars = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for path in input_paths:
            if not path.exists():
                print(f"Пропущено (нет файла): {path}")
                continue
            print(f"Обработка: {path}")
            for doc in iter_corpus(path):
                title = doc.get("title", "")
                text = _clean(doc.get("text", ""))
                if len(text) < MIN_PARAGRAPH_CHARS:
                    continue
                if any(p in title for p in SKIP_PREFIXES):
                    continue
                for para in paragraphs_from_text(text):
                    if len(para) < MIN_PARAGRAPH_CHARS:
                        continue
                    record = {"title": title, "text": para}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    total_chars += len(para)
                    if max_docs and written >= max_docs:
                        print(f"Достигнут лимит документов: {max_docs}")
                        return written
                    if max_chars and total_chars >= max_chars:
                        print(f"Достигнут лимит символов: {max_chars}")
                        return written
    print(f"Готово. Документов: {written}, символов: {total_chars}")
    return written
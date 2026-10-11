
import bz2
import gzip
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List

from config import Config


MIN_PARAGRAPH_CHARS = 80
MAX_PARAGRAPH_CHARS = 2000

SKIP_PREFIXES = (
    "Википедия:",
    "Шаблон:",
    "Категория:",
    "Файл:",
    "Участник:",
    "Обсуждение:",
    "Wikipedia:",
    "Template:",
    "Category:",
    "File:",
    "User:",
    "Talk:"
)

SKIP_SECTIONS = (
    "см. также",
    "примечания",
    "литература",
    "ссылки",
    "источники",
    "библиография",
    "see also",
    "references",
    "external links",
    "further reading"
)


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"(?is)<!--.*?-->",
        " ",
        text
    )

    text = re.sub(
        r"(?is)<ref\b[^>]*>.*?</ref\s*>",
        " ",
        text
    )

    text = re.sub(
        r"(?is)<ref\b[^>]*/>",
        " ",
        text
    )

    text = re.sub(
        r"(?is)<(?:script|style|math)\b[^>]*>.*?</(?:script|style|math)\s*>",
        " ",
        text
    )

    text = re.sub(
        r"\{\{[^{}]*\}\}",
        " ",
        text
    )

    text = re.sub(
        r"\{\{[^{}]*\}\}",
        " ",
        text
    )

    text = re.sub(
        r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]",
        r"\1",
        text
    )

    text = re.sub(
        r"\[https?://[^\s\]]+\s+([^\]]+)\]",
        r"\1",
        text
    )

    text = re.sub(
        r"\[https?://[^\]]+\]",
        " ",
        text
    )

    text = re.sub(
        r"(?is)<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"'{2,}",
        "",
        text
    )

    text = re.sub(
        r"(?m)^\s*[\*#;:]+\s*",
        "",
        text
    )

    text = text.replace(
        "&nbsp;",
        " "
    )

    text = text.replace(
        "&amp;",
        "&"
    )

    text = text.replace(
        "&quot;",
        '"'
    )

    text = text.replace(
        "&lt;",
        "<"
    )

    text = text.replace(
        "&gt;",
        ">"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _read_page_element(element):
    title = ""
    body = ""
    namespace = "0"

    for child in element:
        name = _local_name(child.tag)

        if name == "title":
            title = child.text or ""

        elif name == "ns":
            namespace = (child.text or "").strip()

        elif name == "revision":
            for revision_child in child:
                if _local_name(revision_child.tag) == "text":
                    body = revision_child.text or ""

    title = title.strip()

    if not title or not body.strip():
        return None

    if namespace != "0":
        return None

    if title.casefold().startswith(
        tuple(prefix.casefold() for prefix in SKIP_PREFIXES)
    ):
        return None

    return {
        "title": title,
        "text": body
    }


def iter_wiki_xml(
    path: Path
) -> Iterator[Dict[str, str]]:
    path = Path(path)

    opener = (
        bz2.open
        if path.name.casefold().endswith(".bz2")
        else gzip.open
        if path.name.casefold().endswith(".gz")
        else open
    )

    try:
        with opener(path, "rb") as stream:
            context = ET.iterparse(
                stream,
                events=("start", "end")
            )

            root = None

            for event, element in context:
                if root is None and event == "start":
                    root = element
                    continue

                if event != "end":
                    continue

                if _local_name(element.tag) != "page":
                    continue

                document = _read_page_element(element)

                if document is not None:
                    yield document

                element.clear()

                if root is not None:
                    root.clear()

    except (
        OSError,
        EOFError,
        ET.ParseError
    ):
        return


def iter_wiki_lines(
    path: Path
) -> Iterator[Dict[str, str]]:
    path = Path(path)

    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace"
        ) as stream:
            title = None

            for line in stream:
                line = line.strip()

                if not line:
                    continue

                if title is None:
                    title = line
                    continue

                yield {
                    "title": title,
                    "text": line
                }

                title = None

    except OSError:
        return


def _iter_paragraph_stream(
    stream,
    title: str
) -> Iterator[Dict[str, str]]:
    buffer = []

    for line in stream:
        line = line.strip()

        if not line:
            if buffer:
                yield {
                    "title": title,
                    "text": " ".join(buffer)
                }

                buffer = []

            continue

        buffer.append(line)

    if buffer:
        yield {
            "title": title,
            "text": " ".join(buffer)
        }


def iter_text(
    path: Path
) -> Iterator[Dict[str, str]]:
    path = Path(path)

    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace"
        ) as stream:
            yield from _iter_paragraph_stream(
                stream,
                path.stem
            )

    except OSError:
        return


def iter_gzip(
    path: Path
) -> Iterator[Dict[str, str]]:
    path = Path(path)

    try:
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace"
        ) as stream:
            yield from _iter_paragraph_stream(
                stream,
                path.stem
            )

    except (OSError, EOFError):
        return


def iter_jsonl(
    path: Path
) -> Iterator[Dict[str, str]]:
    path = Path(path)

    opener = (
        gzip.open
        if path.name.casefold().endswith(".gz")
        else open
    )

    try:
        with opener(
            path,
            "rt",
            encoding="utf-8",
            errors="replace"
        ) as stream:
            for line in stream:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(record, dict):
                    continue

                title = record.get(
                    "title",
                    record.get("id", path.stem)
                )

                text = record.get(
                    "text",
                    record.get("content", "")
                )

                if not isinstance(text, str):
                    continue

                if not text.strip():
                    continue

                yield {
                    "title": str(title),
                    "text": text
                }

    except (OSError, EOFError):
        return


def iter_corpus(
    path: Path
) -> Iterator[Dict[str, str]]:
    path = Path(path)
    name = path.name.casefold()
    suffix = path.suffix.casefold()

    if name.endswith(
        (".xml.bz2", ".xml.gz")
    ):
        yield from iter_wiki_xml(path)

    elif name.endswith(
        (".jsonl.gz", ".ndjson.gz")
    ):
        yield from iter_jsonl(path)

    elif suffix in (".jsonl", ".ndjson"):
        yield from iter_jsonl(path)

    elif suffix == ".xml":
        yield from iter_wiki_xml(path)

    elif suffix == ".bz2":
        yield from iter_wiki_xml(path)

    elif suffix == ".gz":
        yield from iter_gzip(path)

    elif suffix in (".txt", ".md"):
        yield from iter_text(path)

    else:
        yield from iter_text(path)


def paragraphs_from_text(
    text: str
) -> List[str]:
    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    results = []
    buffer = []
    current_length = 0

    def flush():
        nonlocal buffer, current_length

        if not buffer:
            return

        paragraph = " ".join(buffer).strip()

        if (
            MIN_PARAGRAPH_CHARS
            <= len(paragraph)
            <= MAX_PARAGRAPH_CHARS
        ):
            results.append(paragraph)

        buffer = []
        current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > MAX_PARAGRAPH_CHARS:
            flush()

            words = sentence.split()
            chunk = []
            chunk_length = 0

            for word in words:
                extra = len(word) + (
                    1 if chunk else 0
                )

                if (
                    chunk
                    and chunk_length + extra > MAX_PARAGRAPH_CHARS
                ):
                    paragraph = " ".join(chunk)

                    if len(paragraph) >= MIN_PARAGRAPH_CHARS:
                        results.append(paragraph)

                    chunk = []
                    chunk_length = 0

                if len(word) > MAX_PARAGRAPH_CHARS:
                    continue

                chunk.append(word)
                chunk_length += len(word) + (
                    1 if chunk_length else 0
                )

            if chunk:
                paragraph = " ".join(chunk)

                if len(paragraph) >= MIN_PARAGRAPH_CHARS:
                    results.append(paragraph)

            continue

        extra = len(sentence) + (
            1 if buffer else 0
        )

        if (
            buffer
            and current_length + extra > MAX_PARAGRAPH_CHARS
        ):
            flush()

        buffer.append(sentence)
        current_length += extra

        if current_length >= MIN_PARAGRAPH_CHARS:
            flush()

    flush()

    return results


def build_corpus_jsonl(
    input_paths: List[Path],
    output_path: Path,
    max_docs: int = 0,
    max_chars: int = 0
) -> int:
    output_path = Path(output_path)

    paths = [
        Path(path)
        for path in input_paths
    ]

    for path in paths:
        if path.resolve() == output_path.resolve():
            raise ValueError(
                "Входной и выходной файл совпадают"
            )

    max_docs = max(0, int(max_docs))
    max_chars = max(0, int(max_chars))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".corpus_",
        suffix=".tmp",
        dir=str(output_path.parent)
    )

    temporary = Path(temporary_name)

    written = 0
    total_chars = 0

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8"
        ) as output:
            stop = False

            for path in paths:
                if stop:
                    break

                if not path.is_file():
                    continue

                for document in iter_corpus(path):
                    if not isinstance(document, dict):
                        continue

                    title = document.get("title", "")
                    text = document.get("text", "")

                    if not isinstance(title, str):
                        title = str(title)

                    if not isinstance(text, str):
                        continue

                    title = title.strip()

                    if title.casefold().startswith(
                        tuple(
                            prefix.casefold()
                            for prefix in SKIP_PREFIXES
                        )
                    ):
                        continue

                    text = _clean(text)

                    if len(text) < MIN_PARAGRAPH_CHARS:
                        continue

                    for paragraph in paragraphs_from_text(text):
                        if max_docs and written >= max_docs:
                            stop = True
                            break

                        if (
                            max_chars
                            and total_chars + len(paragraph) > max_chars
                        ):
                            stop = True
                            break

                        record = {
                            "title": title,
                            "text": paragraph
                        }

                        output.write(
                            json.dumps(
                                record,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        written += 1
                        total_chars += len(paragraph)

            output.flush()
            os.fsync(output.fileno())

        os.replace(
            temporary,
            output_path
        )

        return written

    finally:
        temporary.unlink(
            missing_ok=True
        )

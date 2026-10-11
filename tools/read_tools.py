import gzip
import zlib
from html.parser import HTMLParser
from urllib.parse import urlparse, quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from typing import List

from config import Config
from tools.security import get_security
from tools.web_tools import decode_ddg_url


USER_AGENTS = [
    Config.WEB_USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


def _encode_url(url: str) -> str:
    parts = urlsplit(url)
    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&%")
    fragment = quote(parts.fragment, safe="%")
    return urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


def _decode_body(raw: bytes, content_encoding: str) -> str:
    enc = (content_encoding or "").lower()
    if "gzip" in enc or raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass
    elif "deflate" in enc:
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            try:
                raw = zlib.decompress(raw, -zlib.MAX_WBITS)
            except zlib.error:
                pass
    return raw.decode("utf-8", errors="replace")


class _TextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript"}
    VOID_TAGS = {
        "meta", "link", "br", "hr", "img", "input", "area",
        "base", "col", "embed", "source", "track", "wbr",
    }

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID_TAGS:
            return
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return " ".join(self.parts)


class ReadTools:
    @staticmethod
    def read_url(url: str, max_chars: int = None) -> str:
        sec = get_security()
        ok, reason = sec.check_network()
        if not ok:
            return reason
        max_chars = max_chars or Config.WEB_MAX_PAGE_CHARS
        url = decode_ddg_url(url)
        url = _encode_url(url)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return f"Недопустимая схема URL: {parsed.scheme}"
        try:
            headers = {
                "User-Agent": USER_AGENTS[0],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": Config.WEB_ACCEPT_LANGUAGE,
                "Accept-Encoding": Config.WEB_ACCEPT_ENCODING,
                "Connection": "close",
            }
            request = Request(url, headers=headers)
            with urlopen(request, timeout=Config.WEB_TIMEOUT_SEC) as response:
                raw = response.read(Config.WEB_MAX_PAGE_BYTES)
                content_encoding = response.headers.get("Content-Encoding", "")
            html = _decode_body(raw, content_encoding)
            parser = _TextExtractor()
            parser.feed(html)
            text = parser.text()
            if not text:
                return "Страница пуста или нечитаема"
            return text[:max_chars]
        except Exception as error:
            return f"Ошибка чтения страницы: {error}"
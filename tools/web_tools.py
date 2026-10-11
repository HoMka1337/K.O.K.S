import time
import random
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, unquote, parse_qs, urlparse
from urllib.request import Request, urlopen
from typing import Dict, List, Optional

from config import Config
from tools.security import get_security


USER_AGENTS = [
    Config.WEB_USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]


class SearchResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._current: Optional[Dict[str, str]] = None
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = str(attributes.get("class") or "")
        if tag == "a" and ("result-link" in classes or "result__a" in classes):
            href = attributes.get("href") or ""
            self._current = {
                "title": "",
                "url": urljoin("https://lite.duckduckgo.com", href),
                "snippet": "",
            }
            self._capture_title = True
        elif self._current and (
            "result-snippet" in classes or "result__snippet" in classes
        ):
            self._capture_snippet = True

    def handle_data(self, data):
        current = self._current
        if current is None:
            return
        if self._capture_title:
            current["title"] += data.strip()
        elif self._capture_snippet:
            current["snippet"] += data.strip()

    def handle_endtag(self, tag):
        current = self._current
        if tag == "a" and self._capture_title:
            self._capture_title = False
            if current is None:
                return
            if current["title"]:
                self.results.append(current)
                self._current = None
        elif self._capture_snippet:
            self._capture_snippet = False


def decode_ddg_url(url: str) -> str:
    if "duckduckgo.com/l/" in url or "/l/?uddg=" in url:
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg:
                return unquote(uddg[0])
        except Exception:
            pass
    return url


class WebTools:
    @staticmethod
    def search(query: str, limit: int = None) -> str:
        limit = limit or Config.WEB_SEARCH_LIMIT
        sec = get_security()
        ok, reason = sec.check_network()
        if not ok:
            return reason
        query = (query or "").strip()
        if not query:
            return "Укажи, что искать в интернете"
        if len(query) > 300:
            query = query[:300]
        low = query.lower()
        if any(m in low for m in ("urlopen", "winerror", "traceback", "поиск не удался")):
            return "Поиск не выполняется для служебных ошибок"

        url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
        last_error = None
        for attempt in range(Config.WEB_RETRY_COUNT):
            try:
                headers = {
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": Config.WEB_ACCEPT_LANGUAGE,
                    "Accept-Encoding": Config.WEB_ACCEPT_ENCODING,
                    "Connection": "close",
                }
                request = Request(url, headers=headers)
                with urlopen(request, timeout=Config.WEB_TIMEOUT_SEC) as response:
                    html = response.read(Config.WEB_MAX_PAGE_BYTES).decode(
                        "utf-8", errors="replace"
                    )
                parser = SearchResultParser()
                parser.feed(html)
                if not parser.results:
                    return "Поиск ничего не нашёл"
                lines = []
                for index, result in enumerate(parser.results[:limit], 1):
                    real_url = decode_ddg_url(result["url"])
                    lines.append(
                        f"{index}. {result['title']}\n{real_url}\n{result['snippet']}"
                    )
                return "\n\n".join(lines)
            except Exception as error:
                last_error = error
                time.sleep(Config.WEB_RETRY_DELAY_SEC + random.random())
        return f"Ошибка интернет-поиска: {last_error}"
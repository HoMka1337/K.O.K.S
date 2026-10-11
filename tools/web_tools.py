
import gzip
import ipaddress
import random
import re
import socket
import time
import zlib
from html.parser import HTMLParser
from typing import Dict, List, Optional
from urllib.parse import (
    parse_qs,
    quote,
    unquote,
    urljoin,
    urlparse,
)
from urllib.request import Request, urlopen

from config import Config
from tools.security import get_security


USER_AGENTS = [
    Config.WEB_USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.3 Safari/605.1.15",
]


class SearchResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._current = None
        self._capture_title = False
        self._capture_snippet = False
        self._snippet_depth = 0

    def _finish(self):
        if self._current is None:
            return

        title = " ".join(
            self._current["title"].split()
        )

        snippet = " ".join(
            self._current["snippet"].split()
        )

        url = decode_ddg_url(
            self._current["url"]
        )

        parsed = urlparse(url)

        if (
            title
            and parsed.scheme in {"http", "https"}
            and parsed.hostname
        ):
            self.results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })

        self._current = None
        self._capture_title = False
        self._capture_snippet = False
        self._snippet_depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = attributes.get("class") or ""

        if tag == "a" and (
            "result-link" in classes
            or "result__a" in classes
        ):
            self._finish()

            href = attributes.get("href") or ""

            self._current = {
                "title": "",
                "url": urljoin(
                    "https://lite.duckduckgo.com",
                    href
                ),
                "snippet": ""
            }

            self._capture_title = True
            return

        if self._current is None:
            return

        if (
            "result-snippet" in classes
            or "result__snippet" in classes
        ):
            self._capture_snippet = True
            self._snippet_depth = 1
            return

        if self._capture_snippet:
            self._snippet_depth += 1

    def handle_data(self, data):
        if self._current is None:
            return

        if self._capture_title:
            self._current["title"] += data

        elif self._capture_snippet:
            self._current["snippet"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self._capture_title:
            self._capture_title = False
            return

        if self._capture_snippet:
            self._snippet_depth -= 1

            if self._snippet_depth <= 0:
                self._capture_snippet = False
                self._snippet_depth = 0
                self._finish()

    def close(self):
        super().close()
        self._finish()


def decode_ddg_url(url: str) -> str:
    if not isinstance(url, str):
        return ""

    url = url.strip()

    try:
        parsed = urlparse(url)

        if (
            parsed.hostname
            and parsed.hostname.casefold().endswith(
                "duckduckgo.com"
            )
            and parsed.path.startswith("/l/")
        ):
            values = parse_qs(
                parsed.query
            ).get("uddg", [])

            if values:
                return values[0]

        return url

    except (TypeError, ValueError):
        return ""


def is_public_url(
    url: str,
    resolve_dns: bool = True
) -> bool:
    if not isinstance(url, str):
        return False

    if len(url) > 4096:
        return False

    try:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return False

        if parsed.username or parsed.password:
            return False

        hostname = parsed.hostname

        if not hostname:
            return False

        hostname = hostname.rstrip(".").casefold()

        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
            or hostname.endswith(".internal")
            or hostname.endswith(".test")
            or hostname.endswith(".invalid")
        ):
            return False

        port = parsed.port

        if port not in (None, 80, 443):
            return False

        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None

        if address is not None:
            return address.is_global

        if not resolve_dns:
            return "." in hostname

        addresses = socket.getaddrinfo(
            hostname,
            port or (
                443 if parsed.scheme == "https" else 80
            ),
            type=socket.SOCK_STREAM
        )

        if not addresses:
            return False

        for entry in addresses:
            ip = ipaddress.ip_address(
                entry[4][0]
            )

            if not ip.is_global:
                return False

        return True

    except (
        OSError,
        ValueError,
        TypeError,
        UnicodeError
    ):
        return False


def decode_web_body(
    data: bytes,
    encoding: str = ""
) -> str:
    encoding = encoding.casefold()

    try:
        if "gzip" in encoding or data.startswith(
            b"\x1f\x8b"
        ):
            data = gzip.decompress(data)

        elif "deflate" in encoding:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                data = zlib.decompress(
                    data,
                    -zlib.MAX_WBITS
                )

    except (OSError, zlib.error, EOFError):
        return ""

    return data.decode(
        "utf-8",
        errors="replace"
    )


class WebTools:
    @staticmethod
    def search(
        query: str,
        limit: int = None
    ) -> str:
        if not getattr(Config, "TOOLS_ENABLE_WEB", True):
            return "Интернет-поиск отключён"

        if not isinstance(query, str):
            return "Некорректный поисковый запрос"

        query = query.strip()

        if not query:
            return "Укажи, что искать в интернете"

        query = query[:300]

        try:
            limit = int(
                limit
                if limit is not None
                else Config.WEB_SEARCH_LIMIT
            )
        except (ValueError, TypeError):
            limit = 5

        limit = max(1, min(limit, 20))

        security = get_security()
        allowed, reason = security.check_network()

        if not allowed:
            return reason

        url = (
            "https://lite.duckduckgo.com/lite/?q="
            + quote(query)
        )

        retry_count = max(
            1,
            min(
                5,
                int(Config.WEB_RETRY_COUNT)
            )
        )

        timeout = max(
            1,
            float(Config.WEB_TIMEOUT_SEC)
        )

        max_bytes = max(
            1024,
            int(Config.WEB_MAX_PAGE_BYTES)
        )

        last_error = None

        for attempt in range(retry_count):
            try:
                headers = {
                    "User-Agent": random.choice(
                        USER_AGENTS
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": Config.WEB_ACCEPT_LANGUAGE,
                    "Accept-Encoding": "identity",
                    "Connection": "close"
                }

                request = Request(
                    url,
                    headers=headers
                )

                with urlopen(
                    request,
                    timeout=timeout
                ) as response:
                    content = response.read(
                        max_bytes + 1
                    )

                    if len(content) > max_bytes:
                        return "Страница поиска слишком большая"

                    encoding = response.headers.get(
                        "Content-Encoding",
                        ""
                    )

                html = decode_web_body(
                    content,
                    encoding
                )

                if not html:
                    return "Не удалось прочитать страницу поиска"

                parser = SearchResultParser()
                parser.feed(html)
                parser.close()

                results = []
                seen = set()

                for result in parser.results:
                    link = result["url"]

                    if not is_public_url(
                        link,
                        resolve_dns=False
                    ):
                        continue

                    if link in seen:
                        continue

                    seen.add(link)
                    results.append(result)

                    if len(results) >= limit:
                        break

                if not results:
                    return "Поиск ничего не нашёл"

                lines = []

                for index, result in enumerate(
                    results,
                    1
                ):
                    lines.append(
                        f"{index}. {result['title']}\n"
                        f"{result['url']}\n"
                        f"{result['snippet']}"
                    )

                output = "\n\n".join(lines)

                max_output = max(
                    100,
                    int(Config.TOOLS_MAX_OUTPUT_CHARS)
                )

                return output[:max_output]

            except Exception as error:
                last_error = error

                if attempt + 1 < retry_count:
                    delay = max(
                        0.0,
                        float(Config.WEB_RETRY_DELAY_SEC)
                    )

                    time.sleep(
                        delay + random.uniform(0.0, 0.5)
                    )

        return (
            "Ошибка интернет-поиска: "
            f"{type(last_error).__name__}: {last_error}"
        )

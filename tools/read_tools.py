
import re
from html.parser import HTMLParser
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from config import Config
from tools.security import get_security
from tools.web_tools import (
    decode_ddg_url,
    decode_web_body,
    is_public_url,
)


USER_AGENTS = [
    Config.WEB_USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


def _encode_url(url: str) -> str:
    parts = urlsplit(url)

    path = quote(
        parts.path,
        safe="/%:@"
    )

    query = quote(
        parts.query,
        safe="=&%/?+:;,@"
    )

    fragment = ""

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        path,
        query,
        fragment
    ))


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, max_redirects=5):
        super().__init__()
        self.max_redirects = max_redirects
        self.redirect_count = 0

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl
    ):
        self.redirect_count += 1

        if self.redirect_count > self.max_redirects:
            raise ValueError(
                "Превышен лимит перенаправлений"
            )

        if not is_public_url(newurl):
            raise ValueError(
                "Перенаправление на запрещённый адрес"
            )

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl
        )


class _TextExtractor(HTMLParser):
    SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "template"
    }

    BREAK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "aside",
        "li",
        "ul",
        "ol",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "tr",
        "table",
        "pre"
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)

        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        if tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1

            return

        if self._skip_depth > 0:
            return

        if tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return

        if not isinstance(data, str):
            return

        text = data.strip()

        if text:
            self.parts.append(
                text + " "
            )

    def text(self) -> str:
        content = "".join(self.parts)

        content = re.sub(
            r"[ \t]+",
            " ",
            content
        )

        content = re.sub(
            r" *\n *",
            "\n",
            content
        )

        content = re.sub(
            r"\n{3,}",
            "\n\n",
            content
        )

        return content.strip()


class ReadTools:
    @staticmethod
    def read_url(
        url: str,
        max_chars: int = None
    ) -> str:
        if not getattr(Config, "TOOLS_ENABLE_WEB", True):
            return "Чтение интернет-страниц отключено"

        if not isinstance(url, str):
            return "Некорректный адрес страницы"

        url = decode_ddg_url(
            url.strip()
        )

        if not url:
            return "Укажи адрес страницы"

        try:
            url = _encode_url(url)

        except (TypeError, ValueError):
            return "Некорректный URL"

        if not is_public_url(url):
            return "Недопустимый или запрещённый адрес страницы"

        try:
            max_chars = int(
                max_chars
                if max_chars is not None
                else Config.WEB_MAX_PAGE_CHARS
            )

        except (TypeError, ValueError):
            max_chars = 30000

        max_chars = max(
            1,
            min(max_chars, 100000)
        )

        security = get_security()
        allowed, reason = security.check_network()

        if not allowed:
            return reason

        try:
            headers = {
                "User-Agent": USER_AGENTS[0],
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                "Accept-Language": Config.WEB_ACCEPT_LANGUAGE,
                "Accept-Encoding": "identity",
                "Connection": "close"
            }

            request = Request(
                url,
                headers=headers
            )

            opener = build_opener(
                _SafeRedirectHandler()
            )

            timeout = max(
                1,
                float(Config.WEB_TIMEOUT_SEC)
            )

            max_bytes = max(
                1024,
                int(Config.WEB_MAX_PAGE_BYTES)
            )

            with opener.open(
                request,
                timeout=timeout
            ) as response:
                content_type = response.headers.get(
                    "Content-Type",
                    ""
                ).casefold()

                allowed_types = (
                    "text/html",
                    "application/xhtml+xml",
                    "text/plain"
                )

                if not any(
                    allowed_type in content_type
                    for allowed_type in allowed_types
                ):
                    return "Неподдерживаемый тип содержимого"

                raw = response.read(
                    max_bytes + 1
                )

                if len(raw) > max_bytes:
                    return "Страница слишком большая"

                encoding = response.headers.get(
                    "Content-Encoding",
                    ""
                )

            decoded = decode_web_body(
                raw,
                encoding
            )

            if not decoded.strip():
                return "Страница пуста или нечитаема"

            if "text/plain" in content_type:
                text = decoded.strip()

            else:
                parser = _TextExtractor()
                parser.feed(decoded)
                parser.close()

                text = parser.text()

            if not text:
                return "Страница пуста или нечитаема"

            return text[:max_chars]

        except Exception as error:
            return (
                "Ошибка чтения страницы: "
                f"{type(error).__name__}: {error}"
            )

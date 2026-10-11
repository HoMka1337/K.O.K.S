
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 65536
MAX_QUERY_LENGTH = 10000


def run_api(
    codex,
    host: str = "127.0.0.1",
    port: int = 8765
) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(
            "API разрешено запускать только на локальном адресе"
        )

    port = int(port)

    if not 1 <= port <= 65535:
        raise ValueError("Недопустимый порт")

    web_root = Path(__file__).resolve().parent / "web"
    request_lock = threading.RLock()
    logger = getattr(
        codex,
        "logger",
        logging.getLogger("koks.api")
    )

    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "KoKsAPI"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def _route(self) -> str:
            return urlsplit(self.path).path

        def _send_bytes(
            self,
            status: int,
            body: bytes,
            content_type: str
        ) -> None:
            self.send_response(status)
            self.send_header(
                "Content-Type",
                content_type
            )
            self.send_header(
                "Content-Length",
                str(len(body))
            )
            self.send_header(
                "Cache-Control",
                "no-store"
            )
            self.send_header(
                "X-Content-Type-Options",
                "nosniff"
            )
            self.send_header(
                "X-Frame-Options",
                "DENY"
            )
            self.send_header(
                "Referrer-Policy",
                "no-referrer"
            )
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "object-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'none'"
            )
            self.end_headers()

            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(
            self,
            status: int,
            payload: dict
        ) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False
            ).encode("utf-8")

            self._send_bytes(
                status,
                body,
                "application/json; charset=utf-8"
            )

        def _origin_allowed(self) -> bool:
            origin = self.headers.get("Origin")

            if not origin:
                return True

            try:
                parsed = urlsplit(origin)
                hostname = parsed.hostname
                origin_port = parsed.port

                return (
                    parsed.scheme == "http"
                    and hostname in {
                        "127.0.0.1",
                        "::1",
                        "localhost"
                    }
                    and origin_port == port
                )

            except (TypeError, ValueError):
                return False

        def _read_json(self):
            content_type = self.headers.get(
                "Content-Type",
                ""
            ).split(";", 1)[0].strip().lower()

            if content_type != "application/json":
                raise TypeError(
                    "Требуется Content-Type: application/json"
                )

            length_header = self.headers.get(
                "Content-Length"
            )

            if length_header is None:
                raise ValueError(
                    "Отсутствует Content-Length"
                )

            try:
                length = int(length_header)
            except (ValueError, TypeError):
                raise ValueError(
                    "Некорректный Content-Length"
                )

            if length <= 0:
                raise ValueError(
                    "Пустое тело запроса"
                )

            if length > MAX_REQUEST_BYTES:
                raise OverflowError(
                    "Слишком большой запрос"
                )

            raw = self.rfile.read(length)

            if len(raw) != length:
                raise ValueError(
                    "Тело запроса получено не полностью"
                )

            try:
                payload = json.loads(
                    raw.decode("utf-8")
                )

            except (
                UnicodeError,
                json.JSONDecodeError
            ):
                raise ValueError(
                    "Некорректный JSON"
                )

            if not isinstance(payload, dict):
                raise ValueError(
                    "Ожидается JSON-объект"
                )

            return payload

        def do_GET(self) -> None:
            route = self._route()

            if route in {"/", "/index.html"}:
                try:
                    page = web_root / "index.html"
                    body = page.read_bytes()

                    self._send_bytes(
                        200,
                        body,
                        "text/html; charset=utf-8"
                    )

                except OSError:
                    self._send_json(
                        404,
                        {
                            "error": "Веб-интерфейс недоступен"
                        }
                    )

                return

            if route == "/health":
                model = getattr(
                    codex,
                    "model",
                    None
                )

                inference = getattr(
                    codex,
                    "inference",
                    None
                )

                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model_loaded": (
                            model is not None
                            and inference is not None
                        ),
                        "generation_ready": bool(
                            model is not None
                            and getattr(
                                model,
                                "generation_ready",
                                False
                            )
                        )
                    }
                )

                return

            if route == "/memory":
                try:
                    with request_lock:
                        context = codex.memory.get_context(
                            limit=20
                        )

                    self._send_json(
                        200,
                        {
                            "context": context
                        }
                    )

                except Exception:
                    logger.exception(
                        "API memory request failed"
                    )

                    self._send_json(
                        500,
                        {
                            "error": "Ошибка чтения памяти"
                        }
                    )

                return

            self._send_json(
                404,
                {
                    "error": "Маршрут не найден"
                }
            )

        def do_POST(self) -> None:
            if self._route() != "/query":
                self.close_connection = True

                self._send_json(
                    404,
                    {
                        "error": "Маршрут не найден"
                    }
                )

                return

            if not self._origin_allowed():
                self.close_connection = True

                self._send_json(
                    403,
                    {
                        "error": "Недопустимый источник запроса"
                    }
                )

                return

            try:
                payload = self._read_json()

                query = payload.get(
                    "query"
                )

                confirmed = payload.get(
                    "confirmed",
                    False
                )

                if not isinstance(query, str):
                    raise ValueError(
                        "Поле query должно быть строкой"
                    )

                query = query.strip()

                if not query:
                    raise ValueError(
                        "Поле query обязательно"
                    )

                if len(query) > MAX_QUERY_LENGTH:
                    raise OverflowError(
                        "Слишком длинный запрос"
                    )

                if not isinstance(confirmed, bool):
                    raise ValueError(
                        "Поле confirmed должно быть логическим"
                    )

                with request_lock:
                    response = codex.run(
                        query,
                        confirmed=confirmed
                    )

                if not isinstance(response, str):
                    response = str(response)

                self._send_json(
                    200,
                    {
                        "response": response
                    }
                )

            except OverflowError as error:
                self.close_connection = True

                self._send_json(
                    413,
                    {
                        "error": str(error)
                    }
                )

            except TypeError as error:
                self.close_connection = True

                self._send_json(
                    415,
                    {
                        "error": str(error)
                    }
                )

            except ValueError as error:
                self.close_connection = True

                self._send_json(
                    400,
                    {
                        "error": str(error)
                    }
                )

            except Exception:
                logger.exception(
                    "API query failed"
                )

                self.close_connection = True

                self._send_json(
                    500,
                    {
                        "error": "Внутренняя ошибка сервера"
                    }
                )

        def log_message(
            self,
            format: str,
            *args
        ) -> None:
            logger.info(
                "api: " + format,
                *args
            )

    class LocalHTTPServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(30)
            return connection, address

    server = LocalHTTPServer(
        (host, port),
        RequestHandler
    )

    print(
        f"KoKs API: http://{host}:{port}"
    )

    print(
        f"Веб-интерфейс: http://{host}:{port}/"
    )

    try:
        server.serve_forever(
            poll_interval=0.5
        )

    except KeyboardInterrupt:
        print("\nAPI остановлен")

    finally:
        server.server_close()

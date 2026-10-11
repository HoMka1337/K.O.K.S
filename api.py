import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict


def run_api(codex, host: str = '127.0.0.1', port: int = 8765) -> None:
    web_root = Path(__file__).parent / 'web'

    class RequestHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self._send_bytes(status, body, 'application/json; charset=utf-8')

        def do_GET(self) -> None:
            if self.path in {'/', '/index.html'}:
                try:
                    body = (web_root / 'index.html').read_bytes()
                    self._send_bytes(200, body, 'text/html; charset=utf-8')
                except OSError as error:
                    self._send_json(500, {'error': f'Интерфейс недоступен: {error}'})
                return
            if self.path == '/health':
                self._send_json(200, {
                    'status': 'ok',
                    'model_loaded': codex.inference is not None,
                })
                return
            if self.path == '/memory':
                self._send_json(200, {'context': codex.memory.get_context(limit=20)})
                return
            self._send_json(404, {'error': 'Маршрут не найден'})

        def do_POST(self) -> None:
            if self.path != '/query':
                self._send_json(404, {'error': 'Маршрут не найден'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
                query = payload.get('query', '')
                confirmed = bool(payload.get('confirmed', False))
                if not isinstance(query, str) or not query.strip():
                    self._send_json(400, {'error': 'Поле query обязательно'})
                    return
                self._send_json(200, {'response': codex.run(query, confirmed=confirmed)})
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {'error': f'Некорректный JSON: {error}'})
            except Exception as error:
                self._send_json(500, {'error': str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            codex.logger.info('api: ' + format, *args)

    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f'KoKs API: http://{host}:{port}')
    print(f'Веб-интерфейс: http://{host}:{port}/')
    print('POST /query {"query": "...", "confirmed": false}')
    print('GET /health, GET /memory')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nAPI остановлен')
    finally:
        server.server_close()
"""Local-only HTTP interface for Pattern Lab; no web framework or cloud calls."""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import webbrowser

import cv2

from pattern_lab import (
    DEFAULT_SEED, DEFAULT_SOURCE_SIZE, MAX_TILE_SIZE, MIN_TILE_SIZE, PREVIEW_EDGE,
    encode_png, generate_patch, get_catalog, repeated_preview, validate_pattern_name,
    validate_seed, variant_metadata,
)

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "templates" / "index.html"
LOGGER = logging.getLogger("pattern_lab")


def _value(query, key, default):
    values = query.get(key)
    return values[0] if values else default


def _integer(query, key, default, minimum, maximum, label):
    try:
        result = int(_value(query, key, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(label + " debe ser un número entero.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} debe estar entre {minimum} y {maximum}.")
    return result


def _params(query):
    name = validate_pattern_name(_value(query, "name", "blocks"))
    seed = validate_seed(_value(query, "seed", str(DEFAULT_SEED)))
    size = _integer(query, "size", DEFAULT_SOURCE_SIZE, 32, 1200, "Tamaño base")
    tile_size = _integer(query, "tile_size", 160, MIN_TILE_SIZE, MAX_TILE_SIZE, "Tamaño del motivo")
    return name, seed, size, tile_size


class PatternLabHandler(BaseHTTPRequestHandler):
    server_version = "PatternLab/0.1"
    sys_version = ""

    def _send(self, body, content_type, status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        if filename:
            safe_name = filename.replace('"', "").replace("\r", "").replace("\n", "")
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, value, status=200, filename=None):
        body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status, filename)

    def _error(self, message, status):
        self._json({"error": message}, status)

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=False)
        path = parsed.path
        try:
            if path == "/":
                if not INDEX_FILE.is_file():
                    self._error("No se encontró templates/index.html.", 500)
                    return
                self._send(INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
                return

            if path == "/api/catalog":
                self._json(get_catalog())
                return

            if path in ("/api/pattern.png", "/api/tile.png", "/api/export.png", "/api/export.json"):
                name, seed, size, tile_size = _params(query)
                patch = generate_patch(name, seed, size)

                if path == "/api/pattern.png":
                    interpolation = cv2.INTER_AREA if tile_size < size else cv2.INTER_LINEAR
                    image = cv2.resize(patch, (tile_size, tile_size), interpolation=interpolation)
                    self._send(encode_png(image), "image/png")
                    return

                if path == "/api/tile.png":
                    self._send(encode_png(repeated_preview(patch, tile_size, PREVIEW_EDGE)), "image/png")
                    return

                if path == "/api/export.png":
                    filename = f"pattern_{name}_seed_{seed}.png"
                    self._send(encode_png(patch), "image/png", filename=filename)
                    return

                metadata = variant_metadata(name, seed, size, tile_size)
                filename = f"pattern_{name}_seed_{seed}.json"
                self._json(metadata, filename=filename)
                return

            if path == "/favicon.ico":
                self._send(b"", "image/x-icon", status=204)
                return

            self._error("Ruta no encontrada.", 404)
        except ValueError as exc:
            self._error(str(exc), 400)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            LOGGER.exception("Error procesando la ruta %s", path)
            self._error("Ocurrió un error interno al generar la vista.", 500)

    def log_message(self, fmt, *args):
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


class PatternLabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Explorador textil local de Pattern Lab.")
    parser.add_argument("--port", type=int, default=8765, help="Puerto local (por defecto: 8765).")
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port debe estar entre 1 y 65535.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        with PatternLabServer(("127.0.0.1", args.port), PatternLabHandler) as server:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            print("Pattern Lab está listo.")
            print("Abrí " + url + " en tu navegador.")
            print("El servidor escucha solo en localhost; no está publicado en Internet.")
            if not args.no_browser:
                try:
                    webbrowser.open(url)
                except Exception:
                    LOGGER.info("No se pudo abrir el navegador automáticamente.")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nPattern Lab detenido.")
    except OSError as exc:
        parser.error(f"No se pudo iniciar el servidor local: {exc}")


if __name__ == "__main__":
    main()

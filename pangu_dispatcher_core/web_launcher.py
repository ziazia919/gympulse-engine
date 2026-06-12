import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from werkzeug.serving import make_server

if not getattr(sys, "frozen", False):
    os.environ.setdefault(
        "GYMPULSE_CSV",
        str(Path(__file__).resolve().parent.parent / "field_dispatch_dataa.csv"),
    )

try:
    from .web_app import app
except ImportError:
    from web_app import app


HOST = "127.0.0.1"
PREFERRED_PORT = 8080


def find_available_port() -> int:
    for port in range(PREFERRED_PORT, PREFERRED_PORT + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available local port found between 8080 and 8099.")


def open_browser_when_ready(url: str):
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url, new=2)
                    return
        except OSError:
            time.sleep(0.1)


def show_error(message: str):
    if sys.platform != "win32":
        print(message, file=sys.stderr)
        return

    import ctypes

    ctypes.windll.user32.MessageBoxW(0, message, "GymPulse WebUI", 0x10)


def main():
    try:
        port = find_available_port()
        url = f"http://{HOST}:{port}"
        server = make_server(HOST, port, app, threaded=True)
        threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()
        server.serve_forever()
    except Exception as exc:
        show_error(f"GymPulse WebUI could not start:\n\n{exc}")
        raise


if __name__ == "__main__":
    main()

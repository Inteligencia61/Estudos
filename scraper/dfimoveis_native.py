"""Transporte local entre o scraper e a extensão do Chrome já conectada."""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

CHANNEL_DIR = Path(__file__).with_name(".chrome-nativo")


class NativeTransportError(RuntimeError):
    pass


def validate_url(url: str) -> str:
    parsed = urlsplit(url)
    allowed_path = parsed.path.startswith(("/venda/df/", "/aluguel/df/", "/imovel/"))
    if (parsed.scheme != "https" or parsed.hostname != "www.dfimoveis.com.br"
            or parsed.port not in (None, 443) or parsed.username or parsed.password or not allowed_path):
        raise NativeTransportError("URL fora do DFImóveis ou do escopo de anúncios.")
    return url


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        deadline = time.monotonic() + 2
        while True:
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NativeTransportError("Mensagem local inválida.")
    return value


@contextmanager
def channel_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise NativeTransportError("Já existe uma conexão ou coleta usando este canal.") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def job_active(job: Path) -> bool:
    try:
        with channel_lock(job / "owner.lock"):
            return False
    except NativeTransportError:
        return True


class ChromeNativeTransport:
    def __init__(self, timeout: int, root: Path = CHANNEL_DIR):
        self.timeout = timeout
        self.root = root
        self.sequence = 0
        self.result = None
        self.job = root / uuid.uuid4().hex

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = channel_lock(self.root / "scraper.lock")
        self.lock.__enter__()
        try:
            self.job.mkdir()
            self.owner_lock = channel_lock(self.job / "owner.lock")
            self.owner_lock.__enter__()
            self.state = {"state": "running", "created": time.time()}
            write_json(self.job / "job.json", self.state)
        except BaseException:
            if hasattr(self, "owner_lock"):
                self.owner_lock.__exit__(None, None, None)
            self.lock.__exit__(None, None, None)
            raise
        return self

    def __call__(self, url: str) -> str:
        validate_url(url)
        self.sequence += 1
        request_id = self.sequence
        response_file = self.job / "response.json"
        write_json(self.job / "request.json", {
            "id": request_id, "url": url, "timeout": self.timeout,
        })
        deadline = time.monotonic() + self.timeout + 10
        try:
            while not response_file.exists():
                if time.monotonic() >= deadline:
                    raise NativeTransportError(
                        "O Chrome não respondeu. Abra o DFImóveis, clique na extensão "
                        "e em Conectar ao scraper. Depois execute novamente."
                    )
                time.sleep(0.2)
            response = read_json(response_file)
            if response.get("id") != request_id or response.get("url") != url:
                raise NativeTransportError("O Chrome respondeu a outra página.")
            if response.get("error"):
                raise NativeTransportError(response["error"])
            html = response.get("html")
            if not isinstance(html, str) or not html.strip():
                raise NativeTransportError("O Chrome retornou uma página vazia.")
            return html
        finally:
            (self.job / "request.json").unlink(missing_ok=True)
            response_file.unlink(missing_ok=True)

    def complete(self, count: int, output: Path) -> None:
        self.result = {"linhas": count, "arquivo": str(output.resolve())}

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.state.update(
                state="done" if self.result is not None and exc_type is None else "failed",
                result=self.result,
            )
            write_json(self.job / "job.json", self.state)
        finally:
            try:
                self.owner_lock.__exit__(exc_type, exc, traceback)
            finally:
                self.lock.__exit__(exc_type, exc, traceback)

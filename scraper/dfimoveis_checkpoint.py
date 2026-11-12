"""Checkpoint transacional das buscas e dos anúncios concluídos."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dfimoveis_native import channel_lock


class CheckpointError(RuntimeError):
    pass


class CollectionCheckpoint:
    def __init__(self, output: Path, settings: dict):
        self.path = output.with_name(output.name + ".checkpoint.sqlite3")
        self.settings = settings

    def __enter__(self):
        self.lock = channel_lock(self.path.with_name(self.path.name + ".lock"))
        self.lock.__enter__()
        self.db = None
        try:
            self.db = sqlite3.connect(self.path)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cursor (
                    id INTEGER PRIMARY KEY CHECK(id=1), task INTEGER NOT NULL,
                    page INTEGER NOT NULL, complete INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS rows (
                    sequence INTEGER PRIMARY KEY, link TEXT UNIQUE NOT NULL, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_page (
                    id INTEGER PRIMARY KEY CHECK(id=1), task INTEGER NOT NULL,
                    page INTEGER NOT NULL, html TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scope_links (
                    task INTEGER NOT NULL, link TEXT NOT NULL, PRIMARY KEY(task,link)
                );
            """)
            stored = self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()
            if stored:
                if json.loads(stored[0]) != self.settings:
                    raise CheckpointError(
                        "Esta saída tem checkpoint de outros filtros ou limites. "
                        "Repita os parâmetros originais ou use outro --saida."
                    )
            else:
                with self.db:
                    self.db.execute("INSERT INTO settings VALUES(1,?)", (json.dumps(self.settings),))
                    self.db.execute("INSERT INTO cursor VALUES(1,0,?,0)", (self.settings["inicio"],))
            return self
        except BaseException:
            if self.db is not None:
                self.db.close()
            self.lock.__exit__(None, None, None)
            raise

    @property
    def position(self) -> tuple[int, int]:
        task, page = self.db.execute("SELECT task,page FROM cursor WHERE id=1").fetchone()
        return task, page

    @property
    def complete(self) -> bool:
        return bool(self.db.execute("SELECT complete FROM cursor WHERE id=1").fetchone()[0])

    def rows(self) -> list[dict[str, str]]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT value FROM rows ORDER BY sequence")]

    def pending(self, task: int, page: int) -> str | None:
        row = self.db.execute(
            "SELECT html FROM pending_page WHERE id=1 AND task=? AND page=?", (task, page)
        ).fetchone()
        return row[0] if row else None

    def scope_links(self, task: int) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT link FROM scope_links WHERE task=?", (task,))}

    def save_page(self, task: int, page: int, html: str, links: set[str]) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO pending_page VALUES(1,?,?,?)", (task, page, html))
            self.db.executemany("INSERT OR IGNORE INTO scope_links VALUES(?,?)", ((task, link) for link in links))

    def save_row(self, row: dict[str, str]) -> None:
        with self.db:
            self.db.execute("INSERT INTO rows(link,value) VALUES(?,?)", (row["link"], json.dumps(row, ensure_ascii=False)))

    def advance(self, task: int, page: int) -> None:
        with self.db:
            self.db.execute("UPDATE cursor SET task=?,page=? WHERE id=1", (task, page))
            self.db.execute("DELETE FROM pending_page")

    def mark_complete(self) -> None:
        with self.db:
            self.db.execute("UPDATE cursor SET complete=1 WHERE id=1")
            self.db.execute("DELETE FROM pending_page")

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.db.close()
        finally:
            self.lock.__exit__(exc_type, exc, traceback)

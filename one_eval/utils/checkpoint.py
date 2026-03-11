import os, shutil, tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

def _copy_sqlite_with_wal(src_db: Path, dst_db: Path):
    # 复制主 db
    if src_db.exists():
        shutil.copy2(src_db, dst_db)

    # 复制 WAL / SHM（如果存在）
    for suffix in ("-wal", "-shm"):
        s = Path(str(src_db) + suffix)
        d = Path(str(dst_db) + suffix)
        if s.exists():
            shutil.copy2(s, d)

@asynccontextmanager
async def get_checkpointer(base_db: Path, mode: str):
    base_db.parent.mkdir(parents=True, exist_ok=True)

    if mode == "run":
        async with AsyncSqliteSaver.from_conn_string(str(base_db)) as cp:
            # Optimize SQLite for concurrency
            # await cp.conn.execute("PRAGMA journal_mode=WAL;")
            await cp.conn.execute("PRAGMA busy_timeout=3000;") # 3s timeout
            yield cp
        return

    # debug：复制到临时文件（注意：用同目录，避免跨盘权限/性能问题）
    fd, tmp_path = tempfile.mkstemp(prefix="eval_debug_", suffix=".db", dir=str(base_db.parent))
    os.close(fd)
    tmp_db = Path(tmp_path)

    try:
        _copy_sqlite_with_wal(base_db, tmp_db)
        async with AsyncSqliteSaver.from_conn_string(str(tmp_db)) as cp:
            # Optimize SQLite for concurrency
            # await cp.conn.execute("PRAGMA journal_mode=WAL;")
            await cp.conn.execute("PRAGMA busy_timeout=3000;")
            yield cp
    finally:
        # 清理临时 db 及其 wal/shm
        for suffix in ("", "-wal", "-shm"):
            Path(str(tmp_db) + suffix).unlink(missing_ok=True)

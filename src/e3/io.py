"""Single-writer, same-directory atomic publication with durable payloads."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile


def atomic_write(path: Path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w+b") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return {"directory_fsync": True}
        except OSError:
            return {"directory_fsync": False}
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return atomic_write(path, lambda stream: stream.write(raw))


@contextmanager
def writer_lock(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise ValueError("Existing writer lock; stale removal requires operator review") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(str(os.getpid()).encode()); stream.flush(); os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink()

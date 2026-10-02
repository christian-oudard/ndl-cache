"""
A lock beside the cache file, shared by every process that uses it.

DuckDB lets one process at a time open a file for writing and refuses the
rest outright, so processes take turns on this lock and keep the file open
only while they read or write it, never across a provider call.
"""
import asyncio
import os
import sys

POLL_SECONDS = 0.02

if sys.platform == 'win32':
    import msvcrt

    def _try_lock(fd: int, shared: bool) -> bool:
        # Windows has no shared lock here, so readers take turns as well.
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fd: int, shared: bool) -> bool:
        # flock rather than lockf: flock locks belong to the open file, so
        # two threads of one process exclude each other as well.
        try:
            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
                        | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _unlock(fd: int):
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """
    Exclusive for a process that may write, shared for read-only ones.

    Acquired by polling rather than a blocking call, so a cancelled query
    never leaves a thread behind that takes the lock later and holds it.
    """

    def __init__(self, path: str):
        self.path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    async def acquire(self, shared: bool = False):
        assert self._fd is None, 'already held'
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            while not _try_lock(fd, shared):
                await asyncio.sleep(POLL_SECONDS)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self):
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            _unlock(fd)
        finally:
            os.close(fd)

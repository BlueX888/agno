"""Sync single-memory CRUD must reject an async DB instead of silently no-oping.

``MemoryManager`` accepts ``Union[BaseDb, AsyncBaseDb]``, but its sync
single-memory methods (``add_user_memory``, ``replace_user_memory``,
``delete_user_memory`` and ``clear``) used to call the async DB APIs without
awaiting them. The coroutine was discarded, nothing was written, and the
method still returned a success memory id.

Sibling sync methods (``create_user_memories``, ``clear_user_memories``,
``optimize_memories``, ``update_memory_task``) raise
``ValueError("... is not supported with an async DB ...")``. These tests pin
the same contract for the single-memory methods.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agno.db.base import AsyncBaseDb
from agno.db.schemas import UserMemory
from agno.memory.manager import MemoryManager


@pytest.fixture
def async_db():
    """AsyncBaseDb double that records calls and whether they were awaited."""
    db = MagicMock(spec=AsyncBaseDb)
    db.upsert_user_memory = AsyncMock(return_value=None)
    db.delete_user_memory = AsyncMock(return_value=None)
    db.clear_memories = AsyncMock(return_value=None)
    return db


def test_add_user_memory_rejects_async_db(async_db):
    manager = MemoryManager(db=async_db)

    with pytest.raises(ValueError, match="not supported with an async DB"):
        manager.add_user_memory(UserMemory(memory="Alice likes tea"), user_id="alice")

    # The async API must not be invoked at all: on the buggy path the
    # AsyncMock was called but never awaited, so the write was lost.
    async_db.upsert_user_memory.assert_not_called()
    assert async_db.upsert_user_memory.await_count == 0


def test_replace_user_memory_rejects_async_db(async_db):
    manager = MemoryManager(db=async_db)

    with pytest.raises(ValueError, match="not supported with an async DB"):
        manager.replace_user_memory("mem-1", UserMemory(memory="Alice likes coffee"), user_id="alice")

    async_db.upsert_user_memory.assert_not_called()
    assert async_db.upsert_user_memory.await_count == 0


def test_delete_user_memory_rejects_async_db(async_db):
    manager = MemoryManager(db=async_db)

    with pytest.raises(ValueError, match="not supported with an async DB"):
        manager.delete_user_memory("mem-1", user_id="alice")

    async_db.delete_user_memory.assert_not_called()
    assert async_db.delete_user_memory.await_count == 0


def test_clear_rejects_async_db(async_db):
    manager = MemoryManager(db=async_db)

    with pytest.raises(ValueError, match="not supported with an async DB"):
        manager.clear()

    async_db.clear_memories.assert_not_called()
    assert async_db.clear_memories.await_count == 0

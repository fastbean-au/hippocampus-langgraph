"""A fake store, so the whole suite runs with no service and no network.

The fake keeps real state rather than scripted answers, because a key-value store is only right if
what it reads back is what it wrote: ids are honoured, a create of a held id fails ALREADY_EXISTS in
its own result, an update replaces body and metadata wholesale, metadata filters are exact match,
and a memory below the minimum significance is dropped. It answers with the client package's own
result types, because half of what this adapter does is read those types correctly.

`forget` is the one thing a real service does that no call here asks for - a consolidation cycle
removing a memory - and it is how the tests prove that a vanished item reads as None.
"""

from __future__ import annotations

import datetime
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional

import grpc
import pytest
from hippocampus import BatchResult, BatchStored, Identity, Memory, Page, SearchMode
from hippocampus.errors import NotFound


class FakeClient:
    """An in-memory Hippocampus, recording what it was asked."""

    def __init__(
        self,
        *,
        search_modes: Optional[List[SearchMode]] = None,
        minimum_significance: int = 0,
    ) -> None:
        self.search_modes = [SearchMode.KEYWORD] if search_modes is None else list(search_modes)
        self.minimum_significance = minimum_significance
        self.memories: Dict[str, Memory] = {}
        self.calls: List[str] = []
        self.recalled: List[str] = []
        self.searches: List[Dict[str, Any]] = []

        # Called with the memory id just before a create lands, so a test can stage a race.
        self.before_create = None

    def forget(self, memory_id: str) -> None:
        """What a consolidation cycle does."""

        del self.memories[memory_id]

    def who_am_i(self, **_: Any) -> Identity:
        self.calls.append("who_am_i")

        return Identity(search_modes=list(self.search_modes))

    def store_memories(self, memories: Iterable[Memory], **_: Any) -> BatchStored:
        self.calls.append("store_memories")

        results = []

        for memory in memories:
            if self.before_create is not None:
                self.before_create(memory.id)

            if memory.significance < self.minimum_significance:
                results.append(BatchResult(rejected=True))
            elif memory.id in self.memories:
                results.append(
                    BatchResult(code=grpc.StatusCode.ALREADY_EXISTS.value[0], error="already exists")
                )
            else:
                self.memories[memory.id] = replace(memory, metadata=dict(memory.metadata))
                results.append(BatchResult(id=memory.id))

        return BatchStored(
            results=results,
            stored=sum(1 for result in results if result.id),
            rejected=sum(1 for result in results if result.rejected),
            failed=sum(1 for result in results if result.code),
        )

    def update_memory(self, memory: Memory, **_: Any) -> bool:
        self.calls.append("update_memory")

        held = self.memories.get(memory.id)

        if held is None:
            raise NotFound("memory not found", grpc.StatusCode.NOT_FOUND)

        self.memories[memory.id] = replace(
            held,
            body=memory.body or held.body,
            significance=memory.significance or held.significance,
            metadata=dict(memory.metadata) or held.metadata,
            timestamp=memory.timestamp or held.timestamp,
        )

        return True

    def delete_memories(self, ids: Iterable[str], **_: Any) -> bool:
        self.calls.append("delete_memories")

        ids = list(ids)

        for memory_id in ids:
            self.memories.pop(memory_id, None)

        return True

    def get_memories(
        self,
        *,
        group: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
        ids: Optional[Iterable[str]] = None,
        limit: int = 0,
        offset: int = 0,
        **_: Any,
    ) -> Page:
        self.calls.append("get_memories")

        matched = self._select(group, metadata, ids)
        limit = min(limit or 25, 200)

        return Page(items=matched[offset : offset + limit], total=len(matched))

    def recall_memories(self, ids: Iterable[str], **_: Any) -> Page:
        self.calls.append("recall_memories")

        found = []

        for memory_id in ids:
            memory = self.memories.get(memory_id)

            if memory is None:
                continue

            memory.recall_count += 1
            memory.recalled_at = datetime.datetime.now(datetime.timezone.utc)
            self.recalled.append(memory_id)
            found.append(memory)

        return Page(items=found, total=len(found))

    def search_memories(
        self,
        query: str,
        *,
        limit: int = 0,
        mode: Optional[SearchMode] = None,
        group: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
        reinforce: bool = False,
        offset: int = 0,
        **_: Any,
    ) -> Page:
        self.calls.append("search_memories")
        self.searches.append({"query": query, "mode": mode, "reinforce": reinforce, "offset": offset})

        if offset > 1000:
            raise AssertionError("searched past the service's offset bound")

        terms = query.lower().split()
        matched = [
            memory
            for memory in self._select(group, metadata, None)
            if any(term in memory.body.lower() for term in terms)
        ]
        page = matched[offset : offset + min(limit or 10, 200)]

        if reinforce:
            self.recall_memories([memory.id for memory in page])
            self.calls.pop()

        return Page(items=page, total=len(matched))

    def _select(
        self,
        group: Optional[str],
        metadata: Optional[Mapping[str, str]],
        ids: Optional[Iterable[str]],
    ) -> List[Memory]:
        wanted = set(ids) if ids else None

        return [
            memory
            for memory in self.memories.values()
            if (wanted is None or memory.id in wanted)
            and (not group or memory.group == group)
            and all(memory.metadata.get(k) == v for k, v in (metadata or {}).items())
        ]


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()

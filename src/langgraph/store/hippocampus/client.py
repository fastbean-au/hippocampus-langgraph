"""The narrow view of a store this package is allowed to hold.

`Hippocampus` exposes every RPC the contract declares, including `purge`, `sleep`, `clear` and the
whole transfer surface. This protocol is the statement of which seven of them a LangGraph store may
reach for, and it is the only thing standing between an agent's long-term memory and the destructive
half of that surface - the same line the LlamaIndex block and the event-source bridges draw.

`delete_memories` is on the list, unlike in the LlamaIndex block, because `BaseStore.delete` is part
of the interface being implemented: it is the by-id scalpel, never the by-filter sweep, and only ever
with ids this package derived from a namespace and key the caller named.

It is `runtime_checkable` so that a real `Hippocampus` satisfies it structurally, with no
registration and no subclassing, and so that a test - or an application wrapping the client for its
own metrics, retries or rate limiting - can pass something else that answers the same seven calls.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryClient(Protocol):
    """What `HippocampusStore` calls. `hippocampus.Hippocampus` satisfies it as it is."""

    def who_am_i(self, **kwargs: Any) -> Any:
        """Report what this deployment can serve. Used to resolve the search mode once."""

    def store_memories(self, memories: Iterable[Any], **kwargs: Any) -> Any:
        """Store a batch of new memories, each with its own result."""

    def update_memory(self, memory: Any, **kwargs: Any) -> Any:
        """Replace an existing memory's body and metadata. The second half of an upsert."""

    def delete_memories(self, ids: Iterable[str], **kwargs: Any) -> Any:
        """Delete memories by id."""

    def get_memories(
        self,
        *,
        group: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
        ids: Optional[Iterable[str]] = None,
        limit: int = 0,
        offset: int = 0,
        **kwargs: Any,
    ) -> Any:
        """List memories without reinforcing them."""

    def recall_memories(self, ids: Iterable[str], **kwargs: Any) -> Any:
        """Read memories by id, reinforcing them."""

    def search_memories(
        self,
        query: str,
        *,
        limit: int = 0,
        mode: Optional[Any] = None,
        group: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
        reinforce: bool = False,
        offset: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Search memory content, optionally recalling what is returned."""

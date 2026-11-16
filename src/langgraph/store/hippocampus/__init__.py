"""Hippocampus as a LangGraph long-term memory store.

    from hippocampus import Hippocampus
    from langgraph.store.hippocampus import HippocampusStore

    store = HippocampusStore(Hippocampus("localhost:50051"), group="support-bot")
    graph = builder.compile(store=store)

What the pairing is for: LangGraph's `BaseStore` is a namespaced key-value store with search, and a
read that refreshes an item is a Hippocampus recall - so an item the agent keeps reading is
reinforced, and one it never reads again decays out of the store on its own. There is no TTL and no
eviction in this package because the store's consolidation cycle is both.
"""

from langgraph.store.hippocampus._version import __version__
from langgraph.store.hippocampus.client import MemoryClient
from langgraph.store.hippocampus.store import (
    DEFAULT_SIGNIFICANCE,
    HippocampusStore,
    StoreWriteError,
)

__all__ = [
    "__version__",
    "DEFAULT_SIGNIFICANCE",
    "HippocampusStore",
    "MemoryClient",
    "StoreWriteError",
]

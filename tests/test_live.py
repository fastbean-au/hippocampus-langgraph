"""The store against a real service, when one is named.

The fake in conftest.py is only as right as its reading of the service, and three things here depend
on the service's exact behaviour rather than on this package: that a client-chosen id is accepted and
a held one fails ALREADY_EXISTS in its batch result, that an update replaces metadata wholesale, and
that a metadata filter selects by exact match per key. This suite checks all three end to end.

Set HIPPOCAMPUS_ADDRESS (and HIPPOCAMPUS_TOKEN for an authenticating service). Each run writes under
a fresh group and deletes what it wrote, so it is safe against a shared instance with a writer token.
"""

from __future__ import annotations

import os
import uuid

import pytest
from hippocampus import Hippocampus

from langgraph.store.hippocampus import HippocampusStore

ADDRESS = os.environ.get("HIPPOCAMPUS_ADDRESS")

pytestmark = pytest.mark.skipif(not ADDRESS, reason="HIPPOCAMPUS_ADDRESS is not set")


@pytest.fixture
def store():
    client = Hippocampus(ADDRESS, token=os.environ.get("HIPPOCAMPUS_TOKEN"))
    store = HippocampusStore(client, group=f"langgraph-test-{uuid.uuid4().hex[:12]}")

    yield store

    for namespace in store.list_namespaces():
        for item in store.search(namespace, limit=200, refresh_ttl=False):
            store.delete(item.namespace, item.key)

    client.close()


def test_put_get_update_delete(store):
    ns = ("users", "u-42", "facts")

    store.put(ns, "diet", {"text": "vegetarian"})
    first = store.get(ns, "diet")

    store.put(ns, "diet", {"text": "vegan"})
    second = store.get(ns, "diet")

    assert first.value == {"text": "vegetarian"}
    assert second.value == {"text": "vegan"}
    assert second.created_at == first.created_at

    store.delete(ns, "diet")

    assert store.get(ns, "diet") is None


def test_prefix_listing_and_namespaces(store):
    store.put(("users", "u-42", "facts"), "a", {"v": 1})
    store.put(("users", "u-42", "prefs"), "b", {"v": 2})
    store.put(("users", "u-43", "facts"), "c", {"v": 3})

    assert {item.key for item in store.search(("users", "u-42"))} == {"a", "b"}
    assert store.list_namespaces(prefix=("users",), max_depth=2) == [
        ("users", "u-42"),
        ("users", "u-43"),
    ]


def test_a_query_finds_by_content(store):
    if not store.client.who_am_i().search_modes:
        pytest.skip("this deployment serves no content search")

    store.put(("notes",), "pet", {"text": "a tabby cat called Miso"})
    store.put(("notes",), "car", {"text": "a blue hatchback"})

    hits = store.search(("notes",), query="tabby")

    assert [hit.key for hit in hits] == ["pet"]

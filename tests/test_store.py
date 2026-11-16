"""The store against the fake: the BaseStore contract, and where Hippocampus deliberately departs."""

from __future__ import annotations

import pytest
from hippocampus import SearchMode
from langgraph.store.base import GetOp, InvalidNamespaceError, PutOp, SearchOp

from langgraph.store.hippocampus import HippocampusStore, StoreWriteError
from langgraph.store.hippocampus import _encoding

from conftest import FakeClient

NS = ("users", "u-42", "facts")


@pytest.fixture
def store(client: FakeClient) -> HippocampusStore:
    return HippocampusStore(client, group="agent")


# ------------------------------------------------------------------------------- the KV contract


def test_put_then_get_round_trips_the_value(store, client):
    store.put(NS, "diet", {"text": "vegetarian", "since": 2019})

    item = store.get(NS, "diet")

    assert item is not None
    assert item.namespace == NS
    assert item.key == "diet"
    assert item.value == {"text": "vegetarian", "since": 2019}
    assert item.created_at.tzinfo is not None


def test_a_put_is_one_memory_in_the_stores_group(store, client):
    store.put(NS, "diet", {"text": "vegetarian"})

    (memory,) = client.memories.values()

    assert memory.id == _encoding.memory_id("agent", NS, "diet")
    assert memory.group == "agent"
    assert memory.metadata["lg.ns.0"] == "users"
    assert memory.metadata["lg.ns.2"] == "facts"
    assert memory.metadata["lg.key"] == "diet"


def test_a_second_put_updates_in_place_and_keeps_created_at(store, client):
    store.put(NS, "diet", {"text": "vegetarian"})
    first = store.get(NS, "diet", refresh_ttl=False)

    store.put(NS, "diet", {"text": "vegan"})
    second = store.get(NS, "diet", refresh_ttl=False)

    assert len(client.memories) == 1
    assert second.value == {"text": "vegan"}
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert "update_memory" in client.calls


def test_get_of_an_unwritten_key_is_none(store):
    assert store.get(NS, "never") is None


def test_delete_removes_and_is_idempotent(store, client):
    store.put(NS, "diet", {"text": "vegetarian"})

    store.delete(NS, "diet")
    store.delete(NS, "diet")

    assert store.get(NS, "diet") is None
    assert client.memories == {}


def test_namespaces_and_groups_address_different_items(client):
    one = HippocampusStore(client, group="a")
    other = HippocampusStore(client, group="b")

    one.put(NS, "k", {"v": 1})
    one.put(("users", "u-43", "facts"), "k", {"v": 2})
    other.put(NS, "k", {"v": 3})

    assert one.get(NS, "k").value == {"v": 1}
    assert one.get(("users", "u-43", "facts"), "k").value == {"v": 2}
    assert other.get(NS, "k").value == {"v": 3}
    assert len(client.memories) == 3


def test_a_batch_runs_in_order_and_the_last_put_wins(store, client):
    results = store.batch(
        [
            PutOp(NS, "k", {"v": 1}),
            PutOp(NS, "k", {"v": 2}),
            GetOp(NS, "k"),
            PutOp(NS, "k", None),
            GetOp(NS, "k"),
        ]
    )

    assert results[0] is None and results[1] is None
    assert results[2].value == {"v": 2}
    assert results[4] is None

    # The two puts coalesced into one write, so there was never a create-then-update.
    assert client.calls.count("store_memories") == 1


async def test_the_async_surface_matches(store):
    await store.aput(NS, "diet", {"text": "vegetarian"})

    item = await store.aget(NS, "diet")
    found = await store.asearch(("users",))

    assert item.value == {"text": "vegetarian"}
    assert [hit.key for hit in found] == ["diet"]


# ------------------------------------------------------------------------------------- searching


def test_search_without_a_query_lists_by_prefix(store):
    store.put(NS, "diet", {"text": "vegetarian"})
    store.put(("users", "u-42", "prefs"), "theme", {"text": "dark"})
    store.put(("users", "u-43", "facts"), "diet", {"text": "omnivore"})

    keys = {(hit.namespace, hit.key) for hit in store.search(("users", "u-42"))}

    assert keys == {(NS, "diet"), (("users", "u-42", "prefs"), "theme")}


def test_search_with_a_query_is_a_content_search(store, client):
    store.put(NS, "diet", {"text": "vegetarian"})
    store.put(NS, "pet", {"text": "a cat called Miso"})

    hits = store.search(("users",), query="cat")

    assert [hit.key for hit in hits] == ["pet"]
    assert client.searches[-1]["mode"] is SearchMode.KEYWORD


def test_a_value_filter_is_applied_and_pages_line_up(store):
    for i in range(30):
        store.put(NS, f"k{i:02}", {"n": i, "kind": "even" if i % 2 == 0 else "odd"})

    first = store.search(NS, filter={"kind": "even"}, limit=5)
    second = store.search(NS, filter={"kind": "even"}, limit=5, offset=5)

    assert len(first) == 5 and len(second) == 5
    assert all(hit.value["kind"] == "even" for hit in first + second)
    assert {hit.key for hit in first}.isdisjoint({hit.key for hit in second})


def test_filter_operators_follow_in_memory_store(store):
    store.put(NS, "a", {"score": 3, "tags": ["x"], "meta": {"lang": "en"}})
    store.put(NS, "b", {"score": 7, "tags": ["y"], "meta": {"lang": "fr"}})
    store.put(NS, "c", {"tags": ["x"]})

    def keys(**kwargs):
        return sorted(hit.key for hit in store.search(NS, **kwargs))

    assert keys(filter={"score": {"$gt": 4}}) == ["b"]
    assert keys(filter={"score": {"$lte": 3}}) == ["a"]
    assert keys(filter={"score": {"$ne": 3}}) == ["b", "c"]
    assert keys(filter={"tags": ["x"]}) == ["a", "c"]
    assert keys(filter={"meta": {"lang": "fr"}}) == ["b"]


def test_a_query_on_a_deployment_with_no_search_says_so():
    store = HippocampusStore(FakeClient(search_modes=[]))
    store.put(NS, "k", {"v": 1})

    with pytest.raises(ValueError, match="serves no content search"):
        store.search(NS, query="anything")

    # Without a query nothing needs the content index, so listing still works.
    assert [hit.key for hit in store.search(NS)] == ["k"]


def test_the_mode_resolves_once_to_the_richest_available():
    client = FakeClient(search_modes=[SearchMode.KEYWORD, SearchMode.HYBRID])
    store = HippocampusStore(client)
    store.put(NS, "k", {"v": "word"})

    store.search(NS, query="word")
    store.search(NS, query="word")

    assert client.calls.count("who_am_i") == 1
    assert client.searches[-1]["mode"] is SearchMode.HYBRID


def test_a_mode_the_deployment_cannot_serve_is_refused_by_name():
    store = HippocampusStore(FakeClient(), mode=SearchMode.SEMANTIC)
    store.put(NS, "k", {"v": "word"})

    with pytest.raises(ValueError, match="cannot serve SearchMode.SEMANTIC"):
        store.search(NS, query="word")


def test_memories_written_by_something_else_are_never_items(store, client):
    from hippocampus import Memory

    client.memories["foreign"] = Memory(
        id="foreign", body="a cat", group="agent", metadata={"role": "user"}
    )

    assert store.search(("users",), query="cat") == []
    assert store.list_namespaces() == []


# --------------------------------------------------------------------------- list_namespaces


def test_list_namespaces_matches_prefix_suffix_and_depth(store):
    for namespace in [
        ("users", "u-42", "facts"),
        ("users", "u-42", "prefs"),
        ("users", "u-43", "facts"),
        ("docs", "v1"),
    ]:
        store.put(namespace, "k", {"v": 1})

    assert store.list_namespaces() == [
        ("docs", "v1"),
        ("users", "u-42", "facts"),
        ("users", "u-42", "prefs"),
        ("users", "u-43", "facts"),
    ]
    assert store.list_namespaces(prefix=("users", "*", "facts")) == [
        ("users", "u-42", "facts"),
        ("users", "u-43", "facts"),
    ]
    assert store.list_namespaces(suffix=("prefs",)) == [("users", "u-42", "prefs")]
    assert store.list_namespaces(prefix=("users",), max_depth=2) == [
        ("users", "u-42"),
        ("users", "u-43"),
    ]
    assert store.list_namespaces(limit=1, offset=1) == [("users", "u-42", "facts")]


# ------------------------------------------------------------- where Hippocampus departs, by design


def test_a_get_reinforces_by_default(store, client):
    store.put(NS, "diet", {"text": "vegetarian"})
    memory_id = _encoding.memory_id("agent", NS, "diet")

    store.get(NS, "diet")

    assert client.recalled == [memory_id]
    assert client.memories[memory_id].recall_count == 1


def test_refresh_ttl_false_reads_without_reinforcing(store, client):
    store.put(NS, "diet", {"text": "vegetarian"})

    store.get(NS, "diet", refresh_ttl=False)
    store.search(NS, refresh_ttl=False)

    assert client.recalled == []


def test_reinforce_false_turns_it_off_store_wide(client):
    store = HippocampusStore(client, reinforce=False)
    store.put(NS, "diet", {"text": "vegetarian"})

    store.get(NS, "diet")
    store.search(NS, query="vegetarian")

    assert client.recalled == []


def test_a_filtered_search_reinforces_only_what_it_returns(store, client):
    store.put(NS, "a", {"kind": "keep", "text": "cat"})
    store.put(NS, "b", {"kind": "drop", "text": "cat"})

    hits = store.search(NS, query="cat", filter={"kind": "keep"})

    assert [hit.key for hit in hits] == ["a"]
    assert client.recalled == [_encoding.memory_id("agent", NS, "a")]


def test_a_forgotten_item_reads_as_none(store, client):
    store.put(NS, "diet", {"text": "vegetarian"})
    client.forget(_encoding.memory_id("agent", NS, "diet"))

    assert store.get(NS, "diet") is None
    assert store.search(NS) == []


def test_an_insignificant_put_is_dropped_quietly():
    client = FakeClient(minimum_significance=60)
    store = HippocampusStore(client, significance=10)

    store.put(NS, "diet", {"text": "vegetarian"})

    assert store.get(NS, "diet") is None


def test_significance_fn_ranks_each_put(client):
    store = HippocampusStore(client, significance_fn=lambda ns, key, value: value["weight"])

    store.put(NS, "a", {"weight": 80})

    (memory,) = client.memories.values()

    assert memory.significance == 80


def test_a_ttl_is_refused(store):
    with pytest.raises(NotImplementedError):
        store.put(NS, "k", {"v": 1}, ttl=5)

    with pytest.raises(NotImplementedError, match="decay is the expiry"):
        store.batch([PutOp(NS, "k", {"v": 1}, ttl=5)])


def test_a_create_that_races_another_writer_becomes_an_update(store, client):
    memory_id = _encoding.memory_id("agent", NS, "k")

    def other_writer(created_id):
        if created_id == memory_id and created_id not in client.memories:
            client.memories[created_id] = _record_for(store, {"v": "theirs"})

    client.before_create = other_writer

    store.put(NS, "k", {"v": "ours"})

    assert store.get(NS, "k").value == {"v": "ours"}


def test_an_update_that_races_consolidation_is_written_afresh(store, client):
    store.put(NS, "k", {"v": 1})
    memory_id = _encoding.memory_id("agent", NS, "k")

    original = client.update_memory

    def forgotten_first(memory, **kwargs):
        client.forget(memory_id)

        return original(memory, **kwargs)

    client.update_memory = forgotten_first

    store.put(NS, "k", {"v": 2})

    assert store.get(NS, "k").value == {"v": 2}


def test_a_failed_write_raises_naming_the_item(store, client):
    from hippocampus import BatchResult, BatchStored

    client.store_memories = lambda memories, **_: BatchStored(
        results=[BatchResult(code=3, error="body too long") for _ in memories]
    )

    with pytest.raises(StoreWriteError, match="body too long") as raised:
        store.put(NS, "k", {"v": 1})

    assert list(raised.value.failures) == [(NS, "k")]


# ------------------------------------------------------------------------------------ addressing


@pytest.mark.parametrize("namespace", [("a.b",), ("",), ("a", "\x00")])
def test_invalid_labels_are_refused_even_through_batch(store, namespace):
    with pytest.raises(InvalidNamespaceError):
        store.batch([GetOp(namespace, "k")])

    with pytest.raises(InvalidNamespaceError):
        store.batch([SearchOp(namespace)])


def test_a_namespace_deeper_than_the_metadata_allows_is_refused_by_name(store):
    deep = tuple(f"l{i}" for i in range(_encoding.MAX_NAMESPACE_DEPTH + 1))

    with pytest.raises(ValueError, match="levels deep"):
        store.put(deep, "k", {"v": 1})


def test_ids_are_stable_and_fit_the_service_bound():
    one = _encoding.memory_id("g", NS, "k")

    assert one == _encoding.memory_id("g", NS, "k")
    assert one != _encoding.memory_id("g", NS, "k2")
    assert one != _encoding.memory_id(None, NS, "k")
    assert _encoding.memory_id(None, ("a", "b"), "c") != _encoding.memory_id(None, ("a",), "b\x00c")
    assert len(one) <= 128


def _record_for(store, value):
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)

    return store._record(_encoding.memory_id("agent", NS, "k"), (NS, "k", value), now, now)

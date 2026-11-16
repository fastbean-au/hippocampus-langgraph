# langgraph-store-hippocampus

[![CI](https://github.com/fastbean-au/hippocampus-langgraph/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/fastbean-au/hippocampus-langgraph/actions/workflows/ci.yaml)
![Dependabot](https://img.shields.io/badge/dependabot-enabled-brightgreen)
[![Known Vulnerabilities](https://snyk.io/test/github/fastbean-au/hippocampus-langgraph/badge.svg)](https://snyk.io/test/github/fastbean-au/hippocampus-langgraph)

A [LangGraph](https://langchain-ai.github.io/langgraph/) long-term memory store backed by
[Hippocampus](https://github.com/fastbean-au/hippocampus) — a memory service that stores what
matters and forgets what stops mattering.

```sh
pip install langgraph-store-hippocampus
```

The import name is `langgraph.store.hippocampus`, beside LangGraph's own `langgraph.store.memory`
and `langgraph.store.postgres`.

## Using it

```python
from hippocampus import Hippocampus
from langgraph.store.hippocampus import HippocampusStore

store = HippocampusStore(Hippocampus("localhost:50051"), group="support-bot")

graph = builder.compile(store=store)
```

It is a `BaseStore`, so everything written against one works unchanged — `store.put`, `store.get`,
`store.search`, `store.delete`, `store.list_namespaces`, their `a`-prefixed async forms, and tools
such as LangMem's that take a store:

```python
store.put(("users", "u-42", "facts"), "diet", {"text": "vegetarian since 2019"})

store.get(("users", "u-42", "facts"), "diet").value
# {'text': 'vegetarian since 2019'}

store.search(("users", "u-42"), query="what does the user eat?", limit=3)
```

The client is passed in already connected rather than built from an address, so this package does
not restate the connection options (token, TLS, private CA, client certificate, deadlines) that
`Hippocampus` already documents. The store does not own the channel and never closes it.

| Argument | Default | What it does |
| :------- | :------ | :----------- |
| `client` | required | A connected `hippocampus.Hippocampus`. |
| `group` | `None` | The group stamped on every item and the one reads are restricted to. Under a group-scoped token, name the token's group: it is also part of each item's id. |
| `significance` | `50` | What every item is stored at. Relative to the deployment's own scale — read `significance_levels()` before changing it. |
| `significance_fn` | `None` | `(namespace, key, value) -> int`, consulted on every put, so an update re-ranks the item. |
| `mode` | `None` | The search mode for a query. `None` resolves once to the richest mode the deployment serves (hybrid, then semantic, then keyword). |
| `reinforce` | `True` | Whether a read with `refresh_ttl` recalls what it returns. See below before turning it off. |

## The loop

LangGraph's `refresh_ttl` asks that reading an item extend its life. In Hippocampus that is exactly
what a **recall** does: it resets the memory's decay clock and raises its effective significance
(see [Consolidation](https://github.com/fastbean-au/hippocampus/blob/main/docs/consolidation.md)).
So a `get` or `search` with `refresh_ttl` true — LangGraph's default — is a recall, and an item the
agent keeps reading is the one that survives. One it never reads again decays out on its own.

```text
  agent writes ──> put stores a memory ──> significance decays with age
                                                    │
  agent reads ──> get / search ──> recall resets the clock, raises significance
                                                    │
                                          never read ──> consolidation forgets it
```

That is the entire retention policy. There is no TTL and no eviction in this package, because the
store already has both, and they are driven by what the agent actually did.

## Where it departs from `InMemoryStore`, deliberately

- **Items disappear.** An item can stop existing between a `put` and the next `get`. The `get` then
  returns `None`, exactly as for a key that was never written. Nothing here treats that as an error.
- **There is no TTL.** `supports_ttl` is `False`, so a `put` naming a `ttl` raises
  `NotImplementedError`. Decay is the expiry.
- **Insignificance is not a failure.** A put below the deployment's `memory.minimumSignificance` is
  quietly dropped, and a `get` of it returns `None`. It is logged at debug.
- **`reinforce=False` breaks the loop**, leaving a store that forgets precisely the items the agent
  has been using. It exists for a read-only observer, not for tuning. `refresh_ttl=False` on a
  single read does the same for that read.
- **`index` on a put is ignored.** The service indexes every memory's body itself, and the body is
  the value as JSON, so the JSON keys are indexed tokens too — noise a keyword search tolerates.
  Which kinds of search are available is the deployment's choice (an embedding model and OpenSearch
  for semantic), not the store's.
- **`SearchItem.score` is `None`.** Results come back ranked — relevance blended with significance
  and recall — but the service does not report the score it ranked by.

## How an item is stored

One item is one memory:

| Item | Memory |
| :--- | :----- |
| `(namespace, key)` | `id` — a UUIDv5 of the group, namespace and key, so a `get` is one read by id |
| `value` | `body`, as compact JSON |
| each namespace label | metadata `lg.ns.0`, `lg.ns.1`, … — so a prefix search is an exact-match filter the service evaluates |
| `key` | metadata `lg.key` |
| `created_at` | metadata `lg.created` (UnixNano), kept across updates |
| `updated_at` | the memory's timestamp; a put restarts the decay clock |

Every memory also carries `lg.v=1`, which every read filters on, so a group shared with the
LlamaIndex block or a broker bridge never surfaces their memories as items.

Two limits follow from metadata's bounds: a namespace is at most 29 levels deep, and a label or key
is at most 512 bytes. Both are refused before the call, by name.

**A value filter is evaluated here, not by the service**, because the value is the body. The
namespace prefix is pushed down; the filter then runs over the pages that come back, with
`InMemoryStore`'s operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, nested and list equality).
A filtered search with a query looks at most 1,200 ranked candidates, which is the service's own
search depth. **`list_namespaces` is a scan**: Hippocampus has no namespace index, so it pages through
every item under the longest literal prefix the conditions begin with.

## What a token needs

The store is a writer: it stores, updates, recalls and deletes by id. It names exactly seven client
calls in its `MemoryClient` protocol — `who_am_i`, `store_memories`, `update_memory`,
`delete_memories`, `get_memories`, `recall_memories` and `search_memories` — and that list, held by
a test, is the only thing between an agent's memory and `Purge`. See
[authorisation](https://github.com/fastbean-au/hippocampus/blob/main/docs/configuration.md#authorisation).

## Compatibility

This package tracks the service release in `SERVICE_VERSION`, whose `hippocampus-client` wheel CI
installs. A service release dispatches a bump here, and a merged bump cuts the matching release, so
`langgraph-store-hippocampus X.Y.Z` is the store tested against service `vX.Y.Z`. CI runs against the
`langgraph-checkpoint` floor in `pyproject.toml` as well as the newest release.

**Neither this package nor `hippocampus-client` is on PyPI yet.** Until they are, install the client
wheel from a service release, then this package without its dependency check:

```sh
pip install "https://github.com/fastbean-au/hippocampus/releases/download/v0.53.4/hippocampus_client-0.53.4-py3-none-any.whl"
pip install "langgraph-checkpoint>=2.0.17"
pip install --no-deps "langgraph-store-hippocampus @ git+https://github.com/fastbean-au/hippocampus-langgraph"
```

## Developing

```sh
pip install "https://github.com/fastbean-au/hippocampus/releases/download/$(cat SERVICE_VERSION)/hippocampus_client-$(cut -c2- SERVICE_VERSION)-py3-none-any.whl"
pip install -e '.[dev]'
python -m pytest
```

The suite runs against an in-memory fake that keeps real state, so it needs no service. Set
`HIPPOCAMPUS_ADDRESS` (and `HIPPOCAMPUS_TOKEN` if the service wants one) to run
`tests/test_live.py` against a real instance as well.

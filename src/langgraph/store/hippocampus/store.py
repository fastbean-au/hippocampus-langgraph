"""A LangGraph `BaseStore` backed by Hippocampus.

LangGraph's store is a namespaced key-value store with search, and Hippocampus maps onto it almost
one to one: an item is a memory, a put is a store or an update, a get is a read by id, a search is a
content search or a filtered listing, and a delete is a delete. `_encoding.py` says how an item is
laid out; this file says how each operation is served. What does not map is the point of the pairing:

- **The store forgets.** An item can stop existing between a put and the next get, because a
  consolidation cycle decided it had stopped mattering. A get then returns None, exactly as for a key
  that was never written, and nothing here treats that as an error.
- **A read that refreshes reinforces.** LangGraph's `refresh_ttl` asks that reading an item extend
  its life, which is precisely what a Hippocampus recall does: it resets the decay clock and raises
  the item's effective significance. So a get or search with `refresh_ttl` true - the default - is a
  recall, and an item the agent keeps reading is the one that survives. `reinforce=False` turns that
  off store-wide, for a read-only observer.
- **There is no TTL.** `supports_ttl` is False, so a put naming one is refused. Decay is the TTL, and
  it is driven by use and significance rather than by a number of minutes.
- **Insignificance is not a failure.** A put below the deployment's `memory.minimumSignificance` is
  quietly dropped by the store, and a get of it returns None. It is logged at debug, never raised.

Every call into the service is blocking gRPC, and `abatch` runs `batch` through `asyncio.to_thread`:
the published client is synchronous, and doing that in one place here is better than a second, async
implementation to keep in step with this one.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hippocampus import Memory, SearchMode
from hippocampus.errors import NotFound
from langgraph.store.base import (
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from langgraph.store.hippocampus import _encoding
from langgraph.store.hippocampus.client import MemoryClient

logger = logging.getLogger(__name__)

# Relative, and meaningless on its own: a deployment's own scale is what matters, so
# `significance_levels()` is worth reading before changing it. 50 is the middle of the range the
# LlamaIndex block ranks conversation turns in, so the two share a store without one drowning the other.
DEFAULT_SIGNIFICANCE = 50

# The service's documented caps. Chunking here rather than letting the service refuse means a caller
# writing a large batch never has to know the numbers.
STORE_MEMORIES_LIMIT = 500
IDS_PER_READ = 200
PAGE_SIZE = 200
SEARCH_OFFSET_LIMIT = 1000

# The gRPC code a batch result carries for an id the store already holds.
_ALREADY_EXISTS = 6

# Richest first. `mode=None` resolves to the best thing the deployment actually serves, so an instance
# that gains an embedding model starts doing hybrid retrieval without a code change.
_MODE_PREFERENCE = (SearchMode.HYBRID, SearchMode.SEMANTIC, SearchMode.KEYWORD)

SignificanceFn = Callable[[Tuple[str, ...], str, Mapping[str, Any]], int]


class StoreWriteError(RuntimeError):
    """One or more puts in a batch could not be written. `failures` maps (namespace, key) to why."""

    def __init__(self, failures: Dict[Tuple[Tuple[str, ...], str], str]) -> None:
        self.failures = failures

        detail = "; ".join(f"{ns}/{key}: {why}" for (ns, key), why in failures.items())

        super().__init__(f"{len(failures)} item(s) were not written: {detail}")


class HippocampusStore(BaseStore):
    """Long-term memory for a LangGraph agent, over a Hippocampus store.

        from hippocampus import Hippocampus
        from langgraph.store.hippocampus import HippocampusStore

        store = HippocampusStore(Hippocampus("localhost:50051"), group="support-bot")
        graph = builder.compile(store=store)

    The client is passed in already connected rather than built from an address here, so that this
    class does not restate the connection parameters (token, TLS, private CA, client certificate,
    deadlines) that `Hippocampus` already documents. The store does not own the channel and never
    closes it.
    """

    supports_ttl = False

    def __init__(
        self,
        client: MemoryClient,
        *,
        group: Optional[str] = None,
        significance: int = DEFAULT_SIGNIFICANCE,
        significance_fn: Optional[SignificanceFn] = None,
        mode: Optional[SearchMode] = None,
        reinforce: bool = True,
    ) -> None:
        """
        Args:
            client: a connected `hippocampus.Hippocampus`. Not owned or closed by the store.
            group: the group label stamped on every item and the one reads are restricted to. Under a
                group-scoped token the service stamps its own, and this should name the same group:
                it is also part of each item's derived id.
            significance: what every item is stored at, unless `significance_fn` says otherwise.
            significance_fn: per-item significance from (namespace, key, value). Consulted on every
                put, so an update re-ranks the item.
            mode: the search mode for a query. None resolves once, at the first query, to the richest
                mode the deployment reports it can serve.
            reinforce: whether a read with `refresh_ttl` recalls what it returns. Turning this off
                leaves a store that forgets the items the agent is actually using.
        """

        self.client = client
        self.group = group
        self.significance = significance
        self.significance_fn = significance_fn
        self.mode = mode
        self.reinforce = reinforce

        # Resolved once from who_am_i(), because it is a property of the deployment rather than of a
        # call. Two concurrent first queries may each resolve it; they reach the same answer.
        self._resolved_mode: Optional[SearchMode] = None

    # ----------------------------------------------------------------------------------- batching

    def batch(self, ops: Iterable[Op]) -> List[Result]:
        """Run ops in order, coalescing each run of consecutive puts into one write.

        A run of puts is flushed before any read that follows it, so a get after a put in the same
        batch sees the put. Within a run the last put to an address wins, as it would if they ran one
        at a time.
        """

        ops = list(ops)
        results: List[Result] = [None] * len(ops)
        pending: Dict[Tuple[Tuple[str, ...], str], PutOp] = {}

        for i, op in enumerate(ops):
            _validate(op)

            if isinstance(op, PutOp):
                address = (tuple(op.namespace), str(op.key))

                # Popped first so a repeated address moves to the end, keeping write order.
                pending.pop(address, None)
                pending[address] = op

                continue

            self._flush(pending)
            pending = {}

            if isinstance(op, GetOp):
                results[i] = self._get(op)
            elif isinstance(op, SearchOp):
                results[i] = self._search(op)
            elif isinstance(op, ListNamespacesOp):
                results[i] = self._list_namespaces(op)
            else:
                raise ValueError(f"unknown store operation {type(op).__name__}")

        self._flush(pending)

        return results

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        """`batch`, off the event loop."""

        return await asyncio.to_thread(self.batch, list(ops))

    # ------------------------------------------------------------------------------------ writing

    def _flush(self, puts: Dict[Tuple[Tuple[str, ...], str], PutOp]) -> None:
        """Apply one run of puts: deletes, then creates, then updates of what already exists."""

        if not puts:
            return

        deletes = [self._id(ns, key) for (ns, key), op in puts.items() if op.value is None]
        writes = {
            self._id(ns, key): (ns, key, op.value)
            for (ns, key), op in puts.items()
            if op.value is not None
        }

        for start in range(0, len(deletes), STORE_MEMORIES_LIMIT):
            self._delete(deletes[start : start + STORE_MEMORIES_LIMIT])

        if not writes:
            return

        existing = self._read_by_id(list(writes), reinforce=False)
        now = datetime.datetime.now(datetime.timezone.utc)
        failures: Dict[Tuple[Tuple[str, ...], str], str] = {}

        creates = [memory_id for memory_id in writes if memory_id not in existing]
        updates = [memory_id for memory_id in writes if memory_id in existing]

        for start in range(0, len(creates), STORE_MEMORIES_LIMIT):
            chunk = creates[start : start + STORE_MEMORIES_LIMIT]
            records = [self._record(memory_id, writes[memory_id], now, now) for memory_id in chunk]
            batch = self.client.store_memories(records)

            for memory_id, result in zip(chunk, batch):
                ns, key, _ = writes[memory_id]

                if result.rejected:
                    logger.debug("%s/%s: the store declined the item as insignificant", ns, key)
                elif result.code == _ALREADY_EXISTS:
                    # Another writer created it between the read above and this write. It exists
                    # now, so the put is an update after all.
                    updates.append(memory_id)
                elif result.failed:
                    failures[(ns, key)] = result.error

        for memory_id in updates:
            ns, key, value = writes[memory_id]
            previous = existing.get(memory_id)
            decoded = _encoding.decode(previous) if previous is not None else None

            # The creation time survives an update, which is the one thing about the old record a
            # put keeps; a raced create has no old record to read it from, and is new as of now.
            created_at = decoded[3] if decoded else now

            try:
                self.client.update_memory(self._record(memory_id, (ns, key, value), created_at, now))
            except NotFound:
                # Forgotten between the read and the update. The put still means "this value is
                # current", so it is written afresh rather than lost.
                batch = self.client.store_memories([self._record(memory_id, (ns, key, value), now, now)])

                if batch[0].failed:
                    failures[(ns, key)] = batch[0].error

        if failures:
            raise StoreWriteError(failures)

    def _record(
        self,
        memory_id: str,
        write: Tuple[Tuple[str, ...], str, Mapping[str, Any]],
        created: datetime.datetime,
        now: datetime.datetime,
    ) -> Memory:
        """The memory one put writes.

        The timestamp is set to now on an update as well as a create: a put is fresh content, so it
        restarts the decay clock, which is LangGraph's "the expiration timer refreshes on write".
        """

        ns, key, value = write
        body, metadata = _encoding.encode(ns, key, value, created)

        significance = self.significance

        if self.significance_fn is not None:
            significance = self.significance_fn(ns, key, value)

        return Memory(
            id=memory_id,
            body=body,
            significance=significance,
            group=self.group,
            metadata=metadata,
            timestamp=now,
        )

    def _delete(self, ids: List[str]) -> None:
        """Delete by id. An id already gone - forgotten, or never written - is not an error."""

        try:
            self.client.delete_memories(ids)
        except NotFound:
            pass

    # ------------------------------------------------------------------------------------ reading

    def _get(self, op: GetOp) -> Optional[Item]:
        memory_id = self._id(tuple(op.namespace), str(op.key))
        found = self._read_by_id([memory_id], reinforce=self._reinforces(op.refresh_ttl))

        if memory_id not in found:
            return None

        return _item(found[memory_id], Item)

    def _read_by_id(self, ids: Sequence[str], reinforce: bool) -> Dict[str, Memory]:
        """The memories held under these ids, keyed by id. Missing ids are simply absent."""

        found: Dict[str, Memory] = {}

        for start in range(0, len(ids), IDS_PER_READ):
            chunk = list(ids[start : start + IDS_PER_READ])

            if reinforce:
                page = self.client.recall_memories(chunk)
            else:
                page = self.client.get_memories(ids=chunk, limit=len(chunk))

            for memory in page:
                if _encoding.decode(memory) is not None:
                    found[memory.id] = memory

        return found

    def _search(self, op: SearchOp) -> List[SearchItem]:
        """A content search when there is a query, a filtered listing when there is not.

        The namespace prefix is pushed to the service as a metadata filter. A value filter cannot be,
        because the value is the body, so it is applied here - and then the service's own paging no
        longer lines up with the caller's, so this pages through the matches itself until it has
        `offset + limit` that pass.
        """

        reinforce = self._reinforces(op.refresh_ttl)
        metadata = _encoding.prefix_filter(tuple(op.namespace_prefix))

        if not op.filter:
            memories = self._fetch(op.query, metadata, op.limit, op.offset, reinforce)

            return [_item(memory, SearchItem) for memory in memories]

        wanted = op.offset + op.limit
        matched: List[Memory] = []
        offset = 0

        while len(matched) < wanted:
            page = self._fetch(op.query, metadata, PAGE_SIZE, offset, reinforce=False)

            for memory in page:
                value = _encoding.decode(memory)

                if value is not None and _matches(value[2], op.filter):
                    matched.append(memory)

            offset += PAGE_SIZE

            if len(page) < PAGE_SIZE or (op.query and offset > SEARCH_OFFSET_LIMIT):
                break

        matched = matched[op.offset : wanted]

        # Recalled only once the filter has decided what is returned: a search's own `reinforce`
        # would also have recalled every candidate the filter then dropped.
        if reinforce and matched:
            self.client.recall_memories([memory.id for memory in matched])

        return [_item(memory, SearchItem) for memory in matched]

    def _fetch(
        self,
        query: Optional[str],
        metadata: Dict[str, str],
        limit: int,
        offset: int,
        reinforce: bool,
    ) -> List[Memory]:
        """One page of either a ranked search or a listing, in the store's own order."""

        if query:
            page = self.client.search_memories(
                query,
                limit=limit,
                offset=offset,
                mode=self._mode_for_deployment(),
                group=self.group,
                metadata=metadata,
                reinforce=reinforce,
            )

            return list(page)

        page = list(
            self.client.get_memories(group=self.group, metadata=metadata, limit=limit, offset=offset)
        )

        if reinforce and page:
            self.client.recall_memories([memory.id for memory in page])

        return page

    def _list_namespaces(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        """Every namespace holding an item, matched, truncated and paged here.

        Hippocampus has no namespace index, so this pages through the items themselves - every one
        under the longest literal prefix the conditions start with, which is pushed to the service as
        a filter. On a store holding many items under an unconstrained prefix it is a scan, and is
        priced like one.
        """

        metadata = _encoding.prefix_filter(_literal_prefix(op.match_conditions or ()))
        namespaces = set()
        offset = 0

        while True:
            page = list(
                self.client.get_memories(
                    group=self.group, metadata=metadata, limit=PAGE_SIZE, offset=offset
                )
            )

            for memory in page:
                namespace = _encoding.namespace_of(memory.metadata)

                if all(_namespace_matches(condition, namespace) for condition in op.match_conditions or ()):
                    namespaces.add(namespace if op.max_depth is None else namespace[: op.max_depth])

            offset += PAGE_SIZE

            if len(page) < PAGE_SIZE:
                break

        return sorted(namespaces)[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------------------------ helpers

    def _id(self, namespace: Tuple[str, ...], key: str) -> str:
        return _encoding.memory_id(self.group, namespace, key)

    def _reinforces(self, refresh_ttl: bool) -> bool:
        return self.reinforce and bool(refresh_ttl)

    def _mode_for_deployment(self) -> SearchMode:
        """Resolve the search mode once against what this deployment reports it can serve.

        Asking rather than trying is the point: a mode with no backend is refused per call with
        FAILED_PRECONDITION, so a store configured for one would fail on every query for the life of
        the process while looking like a search bug.
        """

        if self._resolved_mode is not None:
            return self._resolved_mode

        available = list(self.client.who_am_i().search_modes)

        if not available:
            raise ValueError(
                "this Hippocampus deployment serves no content search, so a query cannot be "
                "answered - who_am_i() reports an empty search_modes. Search without a query to "
                "list by namespace and filter, or configure a content-search backend on the service."
            )

        if self.mode is not None:
            if self.mode not in available:
                served = ", ".join(mode.name for mode in available)

                raise ValueError(
                    f"this deployment cannot serve SearchMode.{self.mode.name}; it serves {served}. "
                    "Leave mode unset to take the richest one it has."
                )

            self._resolved_mode = self.mode

            return self._resolved_mode

        self._resolved_mode = next(
            (mode for mode in _MODE_PREFERENCE if mode in available), available[0]
        )

        return self._resolved_mode


def _validate(op: Op) -> None:
    """The checks `BaseStore`'s convenience methods make, repeated for ops passed to batch directly.

    A label holding a period would otherwise be accepted and stored, and a NUL would let two
    addresses share a derived id.
    """

    if isinstance(op, (GetOp, PutOp)):
        _check_labels(tuple(op.namespace))
        _encoding.check_address(tuple(op.namespace), str(op.key))
    elif isinstance(op, SearchOp):
        _check_labels(tuple(op.namespace_prefix))
    elif isinstance(op, ListNamespacesOp):
        for condition in op.match_conditions or ():
            _check_labels(tuple(label for label in condition.path if label != "*"))

    if isinstance(op, PutOp):
        if not op.namespace:
            raise InvalidNamespaceError("Namespace cannot be empty.")

        if op.ttl is not None:
            raise NotImplementedError(
                "HippocampusStore does not support TTLs: the store's decay is the expiry, driven by "
                "use and significance. Put with ttl=None."
            )


def _check_labels(namespace: Tuple[str, ...]) -> None:
    for label in namespace:
        if not isinstance(label, str) or not label or "." in label or "\x00" in label:
            raise InvalidNamespaceError(
                f"Invalid namespace label {label!r} in {namespace}: labels are non-empty strings "
                "with no periods or NUL bytes."
            )


def _item(memory: Memory, kind: type) -> Any:
    namespace, key, value, created = _encoding.decode(memory)
    updated = memory.timestamp or created

    return kind(namespace=namespace, key=key, value=value, created_at=created, updated_at=updated)


def _literal_prefix(conditions: Iterable[MatchCondition]) -> Tuple[str, ...]:
    """The longest leading run of literal labels in a prefix condition, if there is one."""

    for condition in conditions:
        if condition.match_type != "prefix":
            continue

        labels = []

        for label in condition.path:
            if label == "*":
                break

            labels.append(label)

        return tuple(labels)

    return ()


def _namespace_matches(condition: MatchCondition, namespace: Tuple[str, ...]) -> bool:
    """LangGraph's own matching rule: element-wise from the chosen end, with '*' matching any label."""

    path = tuple(condition.path)

    if len(namespace) < len(path):
        return False

    if condition.match_type == "suffix":
        namespace, path = namespace[::-1], path[::-1]
    elif condition.match_type != "prefix":
        raise ValueError(f"Unsupported match type: {condition.match_type}")

    return all(want == "*" or have == want for have, want in zip(namespace, path))


def _matches(value: Mapping[str, Any], filter: Mapping[str, Any]) -> bool:
    """Whether an item's value passes a filter, by the rules `InMemoryStore` applies."""

    return all(_compare(value.get(key), want) for key, want in filter.items())


def _compare(have: Any, want: Any) -> bool:
    if isinstance(want, dict):
        if any(key.startswith("$") for key in want):
            return all(_operator(have, key, operand) for key, operand in want.items())

        if not isinstance(have, dict):
            return False

        return all(_compare(have.get(key), sub) for key, sub in want.items())

    if isinstance(want, (list, tuple)):
        return (
            isinstance(have, (list, tuple))
            and len(have) == len(want)
            and all(_compare(h, w) for h, w in zip(have, want))
        )

    return have == want


def _operator(have: Any, operator: str, operand: Any) -> bool:
    if operator == "$eq":
        return have == operand

    if operator == "$ne":
        return have != operand

    comparisons = {
        "$gt": lambda a, b: a > b,
        "$gte": lambda a, b: a >= b,
        "$lt": lambda a, b: a < b,
        "$lte": lambda a, b: a <= b,
    }

    if operator not in comparisons:
        raise ValueError(f"Unsupported operator: {operator}")

    # A missing or non-numeric field fails the comparison rather than raising, so one malformed item
    # cannot fail a search over all the others.
    try:
        return comparisons[operator](float(have), float(operand))
    except (TypeError, ValueError):
        return False

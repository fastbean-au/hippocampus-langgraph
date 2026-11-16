"""How a LangGraph item is written as a Hippocampus memory, and read back.

    namespace ("users", "u-42", "facts"), key "diet", value {"text": "vegetarian"}

    id        uuid5 of (group, namespace, key)
    body      {"text":"vegetarian"}                   the value, as compact JSON
    metadata  lg.v      = 1                           this encoding, and the marker
              lg.ns.0   = users                       one key per namespace level
              lg.ns.1   = u-42
              lg.ns.2   = facts
              lg.key    = diet
              lg.created = 1763270400000000000        UnixNano of the first put

Three decisions carry it:

- **The id is derived, not stored.** A memory id is any string up to 128 bytes, so the address IS
  the id: a get is one read by id rather than a metadata scan, and a put is a create that falls back
  to an update when the id is already held. The group is part of the seed because two groups may use
  the same namespace and key, and under group scoping each must address its own record.
- **One metadata key per namespace level.** The service's metadata filter is exact match only, so a
  prefix search could not be expressed against a single joined namespace string. With a key per
  level, the prefix `("users", "u-42")` is the filter `lg.ns.0=users, lg.ns.1=u-42`, which the
  service evaluates and pages itself.
- **The value is the body.** Metadata values stop at 512 bytes, so the value cannot live there; the
  body is the only field big enough, and it is also what the content index searches. The JSON keys
  are therefore indexed tokens too, which is noise a keyword search tolerates.

`lg.v` doubles as the marker every read filters on, so a group shared with the LlamaIndex block or a
broker bridge never surfaces their memories as LangGraph items. A future change to this layout is a
new value, read alongside the old one.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Dict, Mapping, Optional, Tuple

from hippocampus import Memory

SCHEMA_KEY = "lg.v"
SCHEMA_VERSION = "1"
KEY_KEY = "lg.key"
CREATED_KEY = "lg.created"
LEVEL_PREFIX = "lg.ns."

# The service allows 32 metadata keys on a memory, and three of them are taken by the fixed keys
# above. Refusing a deeper namespace here names the cause; the service would refuse it as "too many
# metadata keys", which names nothing the caller wrote.
MAX_NAMESPACE_DEPTH = 32 - 3

# The service's own bound on a metadata value, which a namespace label and a key both become.
MAX_LABEL_BYTES = 512

# A fixed namespace for uuid5. Changing it re-addresses every item ever written, so it never changes.
_ID_NAMESPACE = uuid.UUID("5d3f0f2e-6b1a-4c43-9a55-7f1f0c6a2b19")


def memory_id(group: Optional[str], namespace: Tuple[str, ...], key: str) -> str:
    """The memory id an item is stored under.

    NUL separates the parts because LangGraph refuses it in a namespace label, so no two different
    addresses can join to the same seed.
    """

    seed = "\x00".join((group or "", *namespace, "", key))

    return str(uuid.uuid5(_ID_NAMESPACE, seed))


def check_address(namespace: Tuple[str, ...], key: str) -> None:
    """Refuse an address the service would refuse, with a reason that names it."""

    if len(namespace) > MAX_NAMESPACE_DEPTH:
        raise ValueError(
            f"namespace {namespace} is {len(namespace)} levels deep; a Hippocampus store holds at "
            f"most {MAX_NAMESPACE_DEPTH}, one metadata key per level"
        )

    for label in (*namespace, key):
        if len(label.encode("utf-8")) > MAX_LABEL_BYTES:
            raise ValueError(
                f"'{label[:40]}...' is longer than {MAX_LABEL_BYTES} bytes, the most a Hippocampus "
                "metadata value can hold"
            )


def prefix_filter(prefix: Tuple[str, ...]) -> Dict[str, str]:
    """The metadata filter selecting every item at or below a namespace prefix."""

    metadata = {SCHEMA_KEY: SCHEMA_VERSION}

    for i, label in enumerate(prefix):
        metadata[f"{LEVEL_PREFIX}{i}"] = label

    return metadata


def encode(
    namespace: Tuple[str, ...],
    key: str,
    value: Mapping[str, Any],
    created: datetime.datetime,
) -> Tuple[str, Dict[str, str]]:
    """The body and metadata an item is written with."""

    metadata = prefix_filter(namespace)
    metadata[KEY_KEY] = key
    metadata[CREATED_KEY] = str(_to_nanos(created))

    # Compact separators keep the stored size honest - the body counts against the store's capacity -
    # and ensure_ascii=False keeps non-Latin text as itself rather than as six-byte escapes.
    body = json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    return body, metadata


def decode(memory: Memory) -> Optional[Tuple[Tuple[str, ...], str, Dict[str, Any], datetime.datetime]]:
    """An item's namespace, key, value and creation time, or None if this memory is not one."""

    metadata = memory.metadata

    if metadata.get(SCHEMA_KEY) != SCHEMA_VERSION or KEY_KEY not in metadata:
        return None

    try:
        value = json.loads(memory.body)
    except ValueError:
        return None

    if not isinstance(value, dict):
        return None

    created = _from_nanos(metadata.get(CREATED_KEY, "")) or memory.timestamp or _now()

    return namespace_of(metadata), metadata[KEY_KEY], value, created


def namespace_of(metadata: Mapping[str, str]) -> Tuple[str, ...]:
    """The namespace the level keys spell, read until the first gap."""

    labels = []

    while f"{LEVEL_PREFIX}{len(labels)}" in metadata:
        labels.append(metadata[f"{LEVEL_PREFIX}{len(labels)}"])

    return tuple(labels)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _to_nanos(moment: datetime.datetime) -> int:
    delta = moment - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _from_nanos(text: str) -> Optional[datetime.datetime]:
    if not text.isdigit():
        return None

    return datetime.datetime.fromtimestamp(int(text) / 1e9, tz=datetime.timezone.utc)

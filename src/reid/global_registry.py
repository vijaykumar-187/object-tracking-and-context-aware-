"""Persistent run-local global identities built only from human confirmation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def load_registry(path: Path | str) -> dict[str, dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Global identity registry must contain a JSON object.")
    return payload


def _write_registry(path: Path, registry: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f"{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(registry, stream, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _member_node(member: dict) -> tuple[str, int]:
    """Return the (camera_id, track_id) identity of one member row."""
    return (str(member.get("camera_id", "")), int(member.get("track_id", -1)))


def _merge_member_rows(current: dict, incoming: dict) -> dict:
    """Combine duplicate rows for one local track without losing SOURCE."""
    merged = dict(current)
    merged.update(incoming)
    if "SOURCE" in (current.get("status"), incoming.get("status")):
        merged["status"] = "SOURCE"
    else:
        merged["status"] = merged.get("status") or "CONFIRMED"
    return merged


def _deduplicate_members(members: list[dict]) -> list[dict]:
    """Keep exactly one row per (camera_id, track_id), sorted deterministically."""
    unique: dict[tuple[str, int], dict] = {}
    for member in members:
        node = _member_node(member)
        if node in unique:
            unique[node] = _merge_member_rows(unique[node], member)
        else:
            unique[node] = dict(member)
    return sorted(unique.values(), key=_member_node)


def _normalise_members(record: dict) -> None:
    """Enforce one row per local track and exactly one SOURCE anchor."""
    members = _deduplicate_members(record.get("members", []))
    record["members"] = members
    if not members:
        record.pop("source", None)
        return
    source = record.get("source") or {}
    anchor_node = (str(source.get("camera_id", "")), int(source.get("track_id", -1)))
    anchor = next((member for member in members
                   if _member_node(member) == anchor_node), None)
    if anchor is None:
        anchor = next((member for member in members
                       if member.get("status") == "SOURCE"), None)
    anchor = anchor or members[0]
    for member in members:
        member["status"] = "SOURCE" if member is anchor else "CONFIRMED"
    record["source"] = {
        "camera_id": anchor["camera_id"],
        "camera_name": anchor.get("camera_name", anchor["camera_id"]),
        "track_id": int(anchor["track_id"]),
    }


def get_or_create_identity(
    path: Path | str,
    camera_id: str,
    track_id: int,
    camera_name: str,
) -> str:
    """Return the existing global ID for a local track, or assign one once."""
    if not camera_id or int(track_id) <= 0:
        raise ValueError("A camera ID and positive local track ID are required.")
    destination = Path(path)
    registry = load_registry(destination)
    node = (str(camera_id), int(track_id))
    for global_id, record in registry.items():
        if any(_member_node(member) == node
               for member in record.get("members", [])):
            return global_id
    numbers = [int(value.removeprefix("GLOBAL-")) for value in registry
               if value.startswith("GLOBAL-") and value.removeprefix("GLOBAL-").isdigit()]
    global_id = f"GLOBAL-{max(numbers, default=0) + 1:03d}"
    registry[global_id] = {
        "global_person_id": global_id,
        "source": {"camera_id": str(camera_id), "camera_name": str(camera_name),
                   "track_id": int(track_id)},
        "members": [{"camera_id": str(camera_id), "camera_name": str(camera_name),
                     "track_id": int(track_id), "status": "SOURCE"}],
    }
    _write_registry(destination, registry)
    return global_id


def ensure_identity_source(path: Path | str, global_person_id: str,
                           camera_id: str, track_id: int,
                           camera_name: str) -> None:
    """Add the selected track to a global ID without moving its SOURCE anchor.

    A new identity is anchored on the supplied track. An existing identity keeps
    its current anchor, and the supplied track joins as a confirmed member. This
    allows several local tracks from one camera to share a single global ID.
    """
    if not camera_id or int(track_id) <= 0:
        raise ValueError("A camera ID and positive local track ID are required.")
    destination = Path(path)
    registry = load_registry(destination)
    node = (str(camera_id), int(track_id))
    other_identities = [
        identity for identity, record in registry.items()
        if identity != global_person_id
        and any(_member_node(member) == node for member in record.get("members", []))
    ]
    if other_identities:
        raise ValueError(
            f"{camera_name} ID {int(track_id)} already belongs to "
            f"{other_identities[0]}. Reject that association before adding it again."
        )
    record = registry.get(global_person_id)
    created = record is None
    if record is None:
        record = {
            "global_person_id": global_person_id,
            "source": {"camera_id": str(camera_id), "camera_name": str(camera_name),
                       "track_id": int(track_id)},
            "members": [],
        }
        registry[global_person_id] = record
    members = record.setdefault("members", [])
    if not any(_member_node(member) == node for member in members):
        members.append({
            "camera_id": str(camera_id), "camera_name": str(camera_name),
            "track_id": int(track_id), "status": "SOURCE" if created else "CONFIRMED",
        })
    _normalise_members(record)
    _write_registry(destination, registry)


def associate_confirmed_track(
    path: Path | str,
    global_person_id: str,
    camera_id: str,
    track_id: int,
    camera_name: str,
    similarity: float,
    model_label: str,
) -> dict:
    """Add a manually confirmed track, merging IDs only after human approval.

    Several local tracks from the same camera may belong to one global ID. The
    one-local-track-one-global-ID invariant is preserved by merging any identity
    that already contains the supplied track.
    """
    if int(track_id) <= 0 or not -1.00001 <= float(similarity) <= 1.00001:
        raise ValueError("Track ID or similarity is outside its valid range.")
    destination = Path(path)
    registry = load_registry(destination)
    if global_person_id not in registry:
        raise KeyError(f"Unknown global person ID: {global_person_id}")
    target = registry[global_person_id]
    node = (str(camera_id), int(track_id))
    assigned = [
        identity for identity, record in registry.items()
        if any(_member_node(member) == node
               for member in record.get("members", []))
    ]
    for previous_id in assigned:
        if previous_id != global_person_id:
            target.setdefault("members", []).extend(
                registry[previous_id].get("members", []))
            if not target.get("source") and registry[previous_id].get("source"):
                target["source"] = registry[previous_id]["source"]
            del registry[previous_id]
    details = {
        "camera_id": str(camera_id), "camera_name": str(camera_name),
        "track_id": int(track_id), "status": "CONFIRMED",
        "similarity": float(similarity), "model_label": str(model_label),
    }
    existing = next((member for member in target.get("members", [])
                     if _member_node(member) == node), None)
    if existing is None:
        target.setdefault("members", []).append(details)
    else:
        preserves_source = existing.get("status") == "SOURCE"
        existing.update(details)
        if preserves_source:
            existing["status"] = "SOURCE"
    _normalise_members(target)
    _write_registry(destination, registry)
    return target


def reject_track_association(
    path: Path | str, global_person_id: str, camera_id: str, track_id: int,
) -> None:
    """Remove a rejected candidate from a global ID, preserving its source."""
    destination = Path(path)
    registry = load_registry(destination)
    record = registry.get(global_person_id)
    if record is None:
        return
    node = (str(camera_id), int(track_id))
    record["members"] = [
        member for member in record.get("members", [])
        if _member_node(member) != node
    ]
    if not record["members"]:
        del registry[global_person_id]
    else:
        _normalise_members(record)
    _write_registry(destination, registry)


def identity_members(path: Path | str, global_person_id: str) -> dict[str, dict]:
    """Get one confirmed/source local track per camera.

    Compatibility accessor: a global ID may hold several tracks from one camera,
    so this returns the preferred member per camera (SOURCE anchor first, then
    highest similarity, then lowest track ID). Use identity_members_all() for
    every member.
    """
    record = load_registry(path).get(global_person_id, {})
    source = record.get("source") or {}
    anchor_node = (str(source.get("camera_id", "")), int(source.get("track_id", -1)))
    chosen: dict[str, dict] = {}
    for member in record.get("members", []):
        camera_id = str(member["camera_id"])
        current = chosen.get(camera_id)
        if current is None or (_member_preference(member, anchor_node)
                               < _member_preference(current, anchor_node)):
            chosen[camera_id] = member
    return chosen


def _member_preference(member: dict,
                       anchor_node: tuple[str, int]) -> tuple[int, int, float, int]:
    """Sort key: recorded anchor first, then SOURCE, similarity, lowest track ID."""
    return (
        0 if _member_node(member) == anchor_node else 1,
        0 if member.get("status") == "SOURCE" else 1,
        -float(member.get("similarity", 0.0) or 0.0),
        int(member.get("track_id", -1)),
    )


def identity_members_all(path: Path | str,
                         global_person_id: str) -> dict[str, list[dict]]:
    """Get every member of one identity, grouped by camera and sorted by track."""
    record = load_registry(path).get(global_person_id, {})
    grouped: dict[str, list[dict]] = {}
    for member in record.get("members", []):
        grouped.setdefault(str(member["camera_id"]), []).append(member)
    for members in grouped.values():
        members.sort(key=lambda member: int(member["track_id"]))
    return grouped


def find_identity_for_track(path: Path | str, camera_id: str, track_id: int) -> str | None:
    """Find an existing run-level identity assigned to a local camera track."""
    node = (str(camera_id), int(track_id))
    for global_id, record in load_registry(path).items():
        if any(_member_node(member) == node
               for member in record.get("members", [])):
            return global_id
    return None

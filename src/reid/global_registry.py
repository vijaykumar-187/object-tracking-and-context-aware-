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
        if any((member["camera_id"], int(member["track_id"])) == node
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
    """Create a supplied stable global ID anchored to its selected source track."""
    destination = Path(path)
    registry = load_registry(destination)
    record = registry.setdefault(global_person_id, {
        "global_person_id": global_person_id,
        "source": {"camera_id": str(camera_id), "camera_name": str(camera_name),
                   "track_id": int(track_id)},
        "members": [],
    })
    node = (str(camera_id), int(track_id))
    if any(member["camera_id"] == camera_id and int(member["track_id"]) != int(track_id)
           for member in record.get("members", [])):
        raise ValueError(
            f"Camera {camera_name} already has another local track in {global_person_id}."
        )
    if not any((member["camera_id"], int(member["track_id"])) == node
               for member in record.get("members", [])):
        record.setdefault("members", []).append({
            "camera_id": str(camera_id), "camera_name": str(camera_name),
            "track_id": int(track_id), "status": "SOURCE",
        })
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
    """Add a manually confirmed track, merging IDs only after human approval."""
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
        if any((member["camera_id"], int(member["track_id"])) == node
               for member in record.get("members", []))
    ]
    for previous_id in assigned:
        if previous_id != global_person_id:
            target["members"].extend(registry[previous_id].get("members", []))
            del registry[previous_id]
    existing = next((member for member in target["members"]
                     if (member["camera_id"], int(member["track_id"])) == node), None)
    conflicting = [member for member in target["members"]
                   if member["camera_id"] == camera_id and int(member["track_id"]) != int(track_id)]
    if conflicting:
        raise ValueError(
            f"{camera_name} already has a different local track assigned to {global_person_id}. "
            "Review or reject that association before adding another."
        )
    details = {
        "camera_id": str(camera_id), "camera_name": str(camera_name),
        "track_id": int(track_id), "status": "CONFIRMED",
        "similarity": float(similarity), "model_label": str(model_label),
    }
    if existing is None:
        target["members"].append(details)
    else:
        existing.update(details)
    target["members"].sort(key=lambda member: (member["camera_id"], int(member["track_id"])))
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
    source = record.get("source", {})
    record["members"] = [
        member for member in record.get("members", [])
        if (member["camera_id"], int(member["track_id"])) != (camera_id, int(track_id))
    ]
    if not record["members"]:
        del registry[global_person_id]
    elif (source.get("camera_id"), int(source.get("track_id", -1))) == (camera_id, int(track_id)):
        new_source = next((member for member in record["members"]
                           if member.get("status") == "SOURCE"), record["members"][0])
        new_source["status"] = "SOURCE"
        record["source"] = {
            "camera_id": new_source["camera_id"],
            "camera_name": new_source.get("camera_name", new_source["camera_id"]),
            "track_id": int(new_source["track_id"]),
        }
    _write_registry(destination, registry)


def identity_members(path: Path | str, global_person_id: str) -> dict[str, dict]:
    """Get this identity's one confirmed/source local track per camera."""
    registry = load_registry(path)
    record = registry.get(global_person_id, {})
    return {member["camera_id"]: member
            for member in record.get("members", [])}


def find_identity_for_track(path: Path | str, camera_id: str, track_id: int) -> str | None:
    """Find an existing run-level identity assigned to a local camera track."""
    node = (str(camera_id), int(track_id))
    for global_id, record in load_registry(path).items():
        if any((member["camera_id"], int(member["track_id"])) == node
               for member in record.get("members", [])):
            return global_id
    return None

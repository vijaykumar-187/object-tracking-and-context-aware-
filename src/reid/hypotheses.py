"""Persist human-reviewed cross-camera identity hypotheses per processing run."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

VALID_STATUSES = {"UNVERIFIED", "SAME PERSON", "DIFFERENT PEOPLE", "UNCERTAIN"}


def confirmed_component(hypotheses: dict[str, dict], camera_id: str,
                        track_id: int) -> set[tuple[str, int]]:
    """Return the selected node and every transitively confirmed SAME PERSON link.

    Nodes are keyed by ``(camera_id, track_id)``, so several local tracks from
    one camera remain distinct members of the same confirmed component.
    """
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for item in hypotheses.values():
        if item.get("status") != "SAME PERSON":
            continue
        a = (str(item.get("camera1_id", "camera1")), int(item["camera1_track_id"]))
        b = (str(item.get("camera2_id", "camera2")), int(item["camera2_track_id"]))
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    source = (str(camera_id), int(track_id))
    visited = {source}
    frontier = [source]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency.get(current, set()) - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return visited


def load_hypotheses(path: Path | str) -> dict[str, dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Cross-camera verification file must contain an object.")
    return payload


def save_hypothesis(path: Path | str, camera1_track_id: int,
                    camera2_track_id: int, similarity: float,
                    model_label: str, status: str,
                    camera1_id: str = "camera1", camera2_id: str = "camera2",
                    camera1_name: str | None = None,
                    camera2_name: str | None = None,
                    global_person_id: str | None = None) -> dict:
    """Atomically update one local-track pair without assigning a global ID."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Status must be one of: {', '.join(sorted(VALID_STATUSES))}.")
    if camera1_track_id <= 0 or camera2_track_id <= 0:
        raise ValueError("Track IDs must be positive.")
    if not -1.00001 <= similarity <= 1.00001:
        raise ValueError("Cosine similarity must be between -1 and 1.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hypotheses = load_hypotheses(destination)
    key = f"{camera1_id}:{camera1_track_id}|{camera2_id}:{camera2_track_id}"
    prior = hypotheses.get(key, {})
    hypothesis_id = prior.get("hypothesis_id")
    if not hypothesis_id:
        hypothesis_id = f"H{len(hypotheses) + 1:03d}"
    hypotheses[key] = {
        "hypothesis_id": hypothesis_id,
        "camera1_track_id": int(camera1_track_id),
        "camera2_track_id": int(camera2_track_id),
        "camera1_id": camera1_id,
        "camera2_id": camera2_id,
        "camera1_name": camera1_name or camera1_id,
        "camera2_name": camera2_name or camera2_id,
        "global_person_id": global_person_id if status == "SAME PERSON" else None,
        "similarity": float(similarity),
        "model_label": str(model_label),
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent,
            prefix=f"{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(hypotheses, stream, indent=2)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return hypotheses[key]


def build_global_identities(hypotheses: dict[str, dict]) -> list[dict]:
    """Derive review-based global groups from manually confirmed pair links.

    These group labels are separate from camera-local tracker IDs. If a group
    contains a pair explicitly marked DIFFERENT PEOPLE, it is flagged for review.
    """
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def root(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = root(parent[node])
        return parent[node]

    edges = []
    different = []
    names: dict[str, str] = {}
    for item in hypotheses.values():
        camera_a = str(item.get("camera1_id", "camera1"))
        camera_b = str(item.get("camera2_id", "camera2"))
        a = (camera_a, int(item["camera1_track_id"]))
        b = (camera_b, int(item["camera2_track_id"]))
        names[camera_a] = str(item.get("camera1_name", camera_a))
        names[camera_b] = str(item.get("camera2_name", camera_b))
        if item.get("status") == "SAME PERSON":
            edges.append((a, b))
        elif item.get("status") == "DIFFERENT PEOPLE":
            different.append((a, b))
    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(parent):
        groups.setdefault(root(node), []).append(node)
    visible = [sorted(members) for members in groups.values() if len(members) > 1]
    visible.sort(key=lambda members: members[0])
    output = []
    for index, members in enumerate(visible, start=1):
        member_set = set(members)
        contradictions = [
            {"camera_a": a[0], "track_a": a[1], "camera_b": b[0], "track_b": b[1]}
            for a, b in different if a in member_set and b in member_set
        ]
        output.append({
            "global_id": f"G{index:03d}",
            "members": [{"camera_id": camera, "camera_name": names.get(camera, camera),
                         "track_id": track} for camera, track in members],
            "review_conflict": bool(contradictions),
            "conflicts": contradictions,
        })
    return output

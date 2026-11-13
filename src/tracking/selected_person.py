"""Pure helpers for user-selected camera-local track controls."""

from __future__ import annotations


def selected_track_key(run_id: str, camera_id: str, track_id: int) -> str:
    """Build a stable UI identifier that cannot collide across cameras/runs."""
    return f"{run_id}:{camera_id}_t{int(track_id):04d}"


def selected_tracking_state(is_selected: bool, is_paused: bool,
                            track_lifecycle: str, stopped: bool = False) -> str:
    """Translate the selected track and the tracker's last evidence into UI state."""
    if stopped:
        return "Stopped"
    if not is_selected:
        return "Not Tracking"
    if is_paused:
        return "Paused"
    return {
        "active at end": "Tracking",
        "temporarily lost": "Temporarily Lost",
        "ended": "Lost",
        "active": "Tracking",
        "lost": "Temporarily Lost",
    }.get(track_lifecycle, "Lost")

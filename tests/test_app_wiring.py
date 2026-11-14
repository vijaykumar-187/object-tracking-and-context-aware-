"""Static wiring tests for the dashboard module and its detector weight guard.

app.py runs Streamlit calls at import time, so these tests inspect its abstract
syntax tree instead of importing it. The identity confirmation sequence used by
the manual review button is replayed against a temporary run folder.
"""

import ast
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detect import load_detector
from src.reid.global_registry import (
    associate_confirmed_track, ensure_identity_source, find_identity_for_track,
    get_or_create_identity, load_registry,
)
from src.reid.hypotheses import load_hypotheses, save_hypothesis

APP_PATH = PROJECT_ROOT / "app.py"
SRC_ROOT = PROJECT_ROOT / "src"


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_names(tree: ast.Module) -> set:
    """Every name app.py binds through an import statement."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


def called_names(tree: ast.Module) -> set:
    """Every bare name app.py calls, such as associate_confirmed_track(...)."""
    return {node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


def locally_defined_names(tree: ast.Module) -> set:
    return {node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def src_function_modules() -> dict:
    """Map each public top-level src/ function to the modules that define it."""
    defined = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        module = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        for node in parse_module(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    defined.setdefault(node.name, set()).add(module)
    return defined


def statement_lists(tree: ast.Module):
    """Yield every list of sibling statements so adjacency can be inspected."""
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value:
                if all(isinstance(item, ast.stmt) for item in value):
                    yield value


def is_subheader(statement: ast.stmt) -> bool:
    call = getattr(statement, "value", None)
    return (isinstance(statement, ast.Expr) and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "subheader")


def is_missing_detector_guard(node: ast.AST) -> bool:
    """Match `if not detector_path.is_file():`."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.UnaryOp):
        return False
    if not isinstance(node.test.op, ast.Not):
        return False
    call = node.test.operand
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "is_file"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "detector_path")


class TestAppHelperImports(unittest.TestCase):
    """Verifies the dashboard imports every project helper it calls."""

    def test_identity_confirmation_helper_is_imported_from_the_registry_module(self):
        tree = parse_module(APP_PATH)
        registry_imports = [node for node in ast.walk(tree)
                            if isinstance(node, ast.ImportFrom)
                            and node.module == "src.reid.global_registry"]
        self.assertEqual(len(registry_imports), 1,
                         "app.py should import the global registry helpers exactly once.")
        names = {alias.name for alias in registry_imports[0].names}
        self.assertIn("associate_confirmed_track", names,
                      "The SAME PERSON review path calls associate_confirmed_track().")

    def test_every_src_helper_called_by_the_app_is_imported(self):
        tree = parse_module(APP_PATH)
        available = imported_names(tree) | locally_defined_names(tree)
        defined = src_function_modules()
        missing = sorted(name for name in called_names(tree) - available if name in defined)
        self.assertEqual(
            missing, [],
            f"app.py calls project helpers it never imports: {missing}",
        )


class TestAppSectionHeaders(unittest.TestCase):
    """Verifies one section heading is rendered per dashboard section."""

    def test_no_section_renders_two_consecutive_subheaders(self):
        duplicates = [first.lineno for statements in statement_lists(parse_module(APP_PATH))
                      for first, second in zip(statements, statements[1:])
                      if is_subheader(first) and is_subheader(second)]
        self.assertEqual(duplicates, [],
                         f"Consecutive subheaders duplicate a heading on lines {duplicates}.")

    def test_camera_workspace_keeps_the_processed_videos_heading(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('st.subheader("Processed camera videos")', source)
        self.assertNotIn('st.subheader("Two-camera video workspace")', source,
                         "The heading duplicated the CAMERA WORKSPACE eyebrow.")


class TestDetectorWeightGuard(unittest.TestCase):
    """Verifies the missing-weights guard fails fast without a dead assignment."""

    def test_missing_weights_raise_a_clear_file_not_found_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary) / "absent_weights.pt"
            with self.assertRaises(FileNotFoundError) as caught:
                load_detector(absent)
        message = str(caught.exception)
        self.assertIn("weights not found", message)
        self.assertIn(str(absent), message,
                      "The error should name the missing weights path.")

    def test_missing_weights_guard_does_not_bind_the_model_variable(self):
        tree = parse_module(APP_PATH)
        guards = [node for node in ast.walk(tree) if is_missing_detector_guard(node)]
        self.assertEqual(len(guards), 1, "app.py should keep exactly one missing-weights guard.")
        guard = guards[0]
        called = {node.func.id for node in ast.walk(guard)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("load_detector", called,
                      "The guard should fail fast through load_detector().")
        assigned = {target.id for node in ast.walk(guard) if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        self.assertNotIn("model", assigned,
                         "Only the cached get_model() call may bind the detector model.")


class TestIdentityConfirmationSequence(unittest.TestCase):
    """Replays the SAME PERSON branch of the manual cross-camera review."""

    def test_confirmed_pair_shares_one_global_id_in_registry_and_hypotheses(self):
        similarity = 0.8123
        model_label = "OSNet x0.25 · MSMT17 Re-ID"
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            registry_path = run_dir / "global_identity_registry.json"
            hypotheses_path = run_dir / "cross_camera_hypotheses.json"

            self.assertIsNone(find_identity_for_track(registry_path, "camera1", 3),
                              "A fresh run has no identity for either track yet.")
            global_id = get_or_create_identity(registry_path, "camera1", 3, "Camera 1")
            ensure_identity_source(registry_path, global_id, "camera1", 3, "Camera 1")
            associate_confirmed_track(registry_path, global_id, "camera2", 7,
                                      "Camera 2", similarity, model_label)
            save_hypothesis(hypotheses_path, 3, 7, similarity, model_label,
                            "SAME PERSON", "camera1", "camera2",
                            "Camera 1", "Camera 2", global_person_id=global_id)

            record = load_registry(registry_path)[global_id]
            members = {(member["camera_id"], int(member["track_id"]))
                       for member in record["members"]}
            self.assertEqual(members, {("camera1", 3), ("camera2", 7)},
                             "Both confirmed local tracks belong to one global identity.")
            self.assertEqual(find_identity_for_track(registry_path, "camera2", 7), global_id)
            confirmed = next(member for member in record["members"]
                             if member["camera_id"] == "camera2")
            self.assertEqual(confirmed["status"], "CONFIRMED")
            self.assertAlmostEqual(confirmed["similarity"], similarity, places=6)

            stored = load_hypotheses(hypotheses_path)["camera1:3|camera2:7"]
            self.assertEqual(stored["status"], "SAME PERSON")
            self.assertEqual(stored["global_person_id"], global_id)
            self.assertAlmostEqual(stored["similarity"], similarity, places=6)


if __name__ == "__main__":
    unittest.main()

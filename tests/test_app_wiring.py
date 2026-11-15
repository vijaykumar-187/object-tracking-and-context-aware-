"""Static wiring tests for the dashboard module and its detector weight guard.

app.py runs Streamlit calls at import time, so these tests inspect its abstract
syntax tree instead of importing it. The identity confirmation sequence used by
the manual review button is replayed against a temporary run folder, and the real
apply_identity_decision() helper is executed with a stub for Streamlit.
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
    get_or_create_identity, identity_members_all, load_registry,
    reject_track_association,
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


class TestSameCameraReviewWiring(unittest.TestCase):
    """Verifies the manual same-camera option inside the matching panel."""

    def test_matching_panel_renders_the_same_camera_expander(self):
        tree = parse_module(APP_PATH)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "show_cross_camera_matching")
        labels = set()
        for node in ast.walk(function):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "expander" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                labels.add(node.args[0].value)
        self.assertIn("Add another local ID from this camera (manual review)", labels)

    def test_both_review_paths_share_one_decision_helper(self):
        tree = parse_module(APP_PATH)
        self.assertIn("apply_identity_decision", locally_defined_names(tree))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "apply_identity_decision"]
        self.assertGreaterEqual(
            len(calls), 2,
            "The cross-camera and same-camera reviews must share the save logic.",
        )

    def test_same_camera_review_starts_at_unverified(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('same_prior = existing_same.get("status", "UNVERIFIED")', source)
        self.assertIn("This score only helps you compare crops. It never assigns an identity.",
                      source)


class TestSameCameraIdentitySequence(unittest.TestCase):
    """Replays the manual same-camera addition performed by the review panel."""

    def _confirm_cross_camera(self, registry_path, hypotheses_path):
        global_id = get_or_create_identity(registry_path, "camera1", 1, "Camera 1")
        ensure_identity_source(registry_path, global_id, "camera1", 1, "Camera 1")
        associate_confirmed_track(registry_path, global_id, "camera2", 8,
                                  "Camera 2", 0.91, "OSNet")
        save_hypothesis(hypotheses_path, 1, 8, 0.91, "OSNet", "SAME PERSON",
                        "camera1", "camera2", "Camera 1", "Camera 2",
                        global_person_id=global_id)
        return global_id

    def _add_same_camera_sibling(self, registry_path, hypotheses_path, global_id,
                                 sibling_id=5, similarity=0.734):
        ensure_identity_source(registry_path, global_id, "camera1", 1, "Camera 1")
        associate_confirmed_track(registry_path, global_id, "camera1", sibling_id,
                                  "Camera 1", similarity, "OSNet")
        save_hypothesis(hypotheses_path, 1, sibling_id, similarity, "OSNet",
                        "SAME PERSON", "camera1", "camera1",
                        "Camera 1", "Camera 1", global_person_id=global_id)
        return similarity

    def test_same_camera_sibling_joins_one_global_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            registry_path = run_dir / "global_identity_registry.json"
            hypotheses_path = run_dir / "cross_camera_hypotheses.json"

            global_id = self._confirm_cross_camera(registry_path, hypotheses_path)
            self.assertIsNone(find_identity_for_track(registry_path, "camera1", 5))
            self._add_same_camera_sibling(registry_path, hypotheses_path, global_id)

            grouped = identity_members_all(registry_path, global_id)
            record = load_registry(registry_path)[global_id]
            hypotheses = load_hypotheses(hypotheses_path)

        self.assertEqual([member["track_id"] for member in grouped["camera1"]], [1, 5])
        self.assertEqual([member["track_id"] for member in grouped["camera2"]], [8])
        self.assertEqual(record["source"],
                         {"camera_id": "camera1", "camera_name": "Camera 1", "track_id": 1})
        self.assertEqual(len([member for member in record["members"]
                              if member["status"] == "SOURCE"]), 1)
        self.assertEqual(set(hypotheses),
                         {"camera1:1|camera2:8", "camera1:1|camera1:5"})
        same_camera = hypotheses["camera1:1|camera1:5"]
        self.assertEqual(same_camera["status"], "SAME PERSON")
        self.assertEqual(same_camera["global_person_id"], global_id)

    def test_same_camera_different_people_removes_only_the_sibling(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            registry_path = run_dir / "global_identity_registry.json"
            hypotheses_path = run_dir / "cross_camera_hypotheses.json"

            global_id = self._confirm_cross_camera(registry_path, hypotheses_path)
            self._add_same_camera_sibling(registry_path, hypotheses_path, global_id)
            reject_track_association(registry_path, global_id, "camera1", 5)
            save_hypothesis(hypotheses_path, 1, 5, 0.734, "OSNet",
                            "DIFFERENT PEOPLE", "camera1", "camera1",
                            "Camera 1", "Camera 1", global_person_id=None)

            rejected = find_identity_for_track(registry_path, "camera1", 5)
            anchor = find_identity_for_track(registry_path, "camera1", 1)
            other_camera = find_identity_for_track(registry_path, "camera2", 8)
            record = load_registry(registry_path)[global_id]
            status = load_hypotheses(hypotheses_path)["camera1:1|camera1:5"]["status"]

        self.assertIsNone(rejected)
        self.assertEqual(anchor, global_id)
        self.assertEqual(other_camera, global_id)
        self.assertEqual(record["source"]["track_id"], 1)
        self.assertEqual(status, "DIFFERENT PEOPLE")


class FakeStreamlit:
    """Minimal stand-in for the Streamlit calls apply_identity_decision() uses."""

    def __init__(self):
        self.session_state = {}
        self.successes = []
        self.errors = []
        self.rerun_count = 0

    def success(self, message):
        self.successes.append(message)

    def error(self, message):
        self.errors.append(message)

    def rerun(self):
        self.rerun_count += 1


def load_identity_decision_helper(streamlit_stub):
    """Execute app.apply_identity_decision() without importing Streamlit."""
    tree = parse_module(APP_PATH)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "apply_identity_decision")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__",
                           names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "Path": Path,
        "st": streamlit_stub,
        "find_identity_for_track": find_identity_for_track,
        "get_or_create_identity": get_or_create_identity,
        "ensure_identity_source": ensure_identity_source,
        "associate_confirmed_track": associate_confirmed_track,
        "reject_track_association": reject_track_association,
        "save_hypothesis": save_hypothesis,
    }
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["apply_identity_decision"]


CAMERA_NAMES = {"camera1": "Camera 1", "camera2": "Camera 2", "camera3": "Camera 3"}


class TestIdentityDecisionTransitions(unittest.TestCase):
    """Executes the real review helper for every manual decision transition.

    A pair can move between SAME PERSON, DIFFERENT PEOPLE, UNVERIFIED, and
    UNCERTAIN through the review panel. Only DIFFERENT PEOPLE may remove a
    registry association that a reviewer already confirmed.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name)
        self.registry_path = self.run_dir / "global_identity_registry.json"
        self.hypotheses_path = self.run_dir / "cross_camera_hypotheses.json"
        self.streamlit = FakeStreamlit()
        self.apply_identity_decision = load_identity_decision_helper(self.streamlit)

    def review(self, decision, source_camera="camera1", source_track=1,
               destination_camera="camera1", destination_track=5,
               similarity=0.42):
        return self.apply_identity_decision(
            self.run_dir, self.registry_path, self.hypotheses_path,
            source_camera, source_track, CAMERA_NAMES[source_camera],
            destination_camera, destination_track,
            CAMERA_NAMES[destination_camera], similarity, "OSNet", decision,
        )

    def registry_members(self):
        return {
            global_id: {(member["camera_id"], int(member["track_id"]))
                        for member in record["members"]}
            for global_id, record in load_registry(self.registry_path).items()
        }

    def confirm_same_camera_pair(self):
        self.review("SAME PERSON", destination_camera="camera2",
                    destination_track=8, similarity=0.91)
        return self.review("SAME PERSON")

    def test_same_person_adds_the_sibling_to_the_confirmed_identity(self):
        global_id = self.confirm_same_camera_pair()

        self.assertIsNotNone(global_id)
        self.assertEqual(
            self.registry_members(),
            {global_id: {("camera1", 1), ("camera1", 5), ("camera2", 8)}},
        )
        record = load_registry(self.registry_path)[global_id]
        self.assertEqual(record["source"]["track_id"], 1)
        hypothesis = load_hypotheses(self.hypotheses_path)["camera1:1|camera1:5"]
        self.assertEqual(hypothesis["status"], "SAME PERSON")
        self.assertEqual(hypothesis["global_person_id"], global_id)
        self.assertEqual(self.streamlit.errors, [])

    def test_unverified_keeps_the_confirmed_registry_association(self):
        global_id = self.confirm_same_camera_pair()
        before = self.registry_members()

        returned = self.review("UNVERIFIED")

        self.assertEqual(self.registry_members(), before)
        for camera, track in [("camera1", 1), ("camera1", 5), ("camera2", 8)]:
            self.assertEqual(
                find_identity_for_track(self.registry_path, camera, track),
                global_id,
            )
        record = load_registry(self.registry_path)[global_id]
        self.assertEqual(record["source"]["track_id"], 1)
        self.assertEqual(
            sum(member["status"] == "SOURCE" for member in record["members"]), 1,
        )
        hypotheses = load_hypotheses(self.hypotheses_path)
        same_camera = hypotheses["camera1:1|camera1:5"]
        self.assertEqual(same_camera["status"], "UNVERIFIED")
        self.assertIsNone(same_camera["global_person_id"])
        self.assertEqual(hypotheses["camera1:1|camera2:8"]["status"], "SAME PERSON")
        self.assertIn("UNVERIFIED", self.streamlit.successes[-1])
        self.assertIsNone(returned)
        self.assertEqual(self.streamlit.errors, [])

    def test_uncertain_keeps_the_confirmed_registry_association(self):
        global_id = self.confirm_same_camera_pair()

        returned = self.review("UNCERTAIN")

        self.assertEqual(
            self.registry_members(),
            {global_id: {("camera1", 1), ("camera1", 5), ("camera2", 8)}},
        )
        hypothesis = load_hypotheses(self.hypotheses_path)["camera1:1|camera1:5"]
        self.assertEqual(hypothesis["status"], "UNCERTAIN")
        self.assertIsNone(hypothesis["global_person_id"])
        self.assertIsNone(returned)
        self.assertEqual(self.streamlit.errors, [])

    def test_different_people_removes_only_the_rejected_sibling(self):
        global_id = self.confirm_same_camera_pair()

        returned = self.review("DIFFERENT PEOPLE")

        self.assertEqual(returned, global_id)
        self.assertIsNone(find_identity_for_track(self.registry_path, "camera1", 5))
        self.assertEqual(
            find_identity_for_track(self.registry_path, "camera1", 1), global_id,
        )
        self.assertEqual(
            find_identity_for_track(self.registry_path, "camera2", 8), global_id,
        )
        record = load_registry(self.registry_path)[global_id]
        self.assertEqual(record["source"]["track_id"], 1)
        self.assertEqual(
            sum(member["status"] == "SOURCE" for member in record["members"]), 1,
        )
        hypothesis = load_hypotheses(self.hypotheses_path)["camera1:1|camera1:5"]
        self.assertEqual(hypothesis["status"], "DIFFERENT PEOPLE")
        self.assertIsNone(hypothesis["global_person_id"])
        self.assertEqual(self.streamlit.errors, [])

    def test_same_person_after_different_people_restores_the_pair(self):
        global_id = self.confirm_same_camera_pair()
        self.review("DIFFERENT PEOPLE")
        self.assertIsNone(find_identity_for_track(self.registry_path, "camera1", 5))

        restored = self.review("SAME PERSON", similarity=0.55)

        self.assertEqual(restored, global_id)
        self.assertEqual(
            self.registry_members(),
            {global_id: {("camera1", 1), ("camera1", 5), ("camera2", 8)}},
        )
        hypothesis = load_hypotheses(self.hypotheses_path)["camera1:1|camera1:5"]
        self.assertEqual(hypothesis["status"], "SAME PERSON")
        self.assertEqual(hypothesis["global_person_id"], global_id)
        self.assertAlmostEqual(hypothesis["similarity"], 0.55, places=6)
        self.assertEqual(self.streamlit.errors, [])

    def test_unverified_does_not_associate_previously_unlinked_tracks(self):
        returned = self.review("UNVERIFIED")

        self.assertIsNone(returned)
        self.assertEqual(load_registry(self.registry_path), {})
        self.assertIsNone(find_identity_for_track(self.registry_path, "camera1", 1))
        self.assertIsNone(find_identity_for_track(self.registry_path, "camera1", 5))
        hypothesis = load_hypotheses(self.hypotheses_path)["camera1:1|camera1:5"]
        self.assertEqual(hypothesis["status"], "UNVERIFIED")
        self.assertIsNone(hypothesis["global_person_id"])

    def test_uncertain_leaves_a_sibling_only_identity_unchanged(self):
        sibling_global_id = get_or_create_identity(
            self.registry_path, "camera1", 5, "Camera 1",
        )
        before = self.registry_members()

        returned = self.review("UNCERTAIN")

        self.assertIsNone(returned)
        self.assertEqual(self.registry_members(), before)
        self.assertEqual(
            find_identity_for_track(self.registry_path, "camera1", 5),
            sibling_global_id,
        )
        self.assertIsNone(find_identity_for_track(self.registry_path, "camera1", 1))


if __name__ == "__main__":
    unittest.main()

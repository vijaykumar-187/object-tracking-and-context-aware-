"""Static Phase D wiring checks for the combined multi-camera video entry point.

app.py runs Streamlit calls at import time and Streamlit is not installed, so the
dashboard is inspected through its abstract syntax tree. Nothing here renders a
video or imports streamlit.
"""

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
VIDEO_RESULTS_PATH = PROJECT_ROOT / "src" / "reid" / "video_results.py"

REGISTRY_WRITERS = {
    "get_or_create_identity", "ensure_identity_source",
    "associate_confirmed_track", "reject_track_association",
}


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def function_node(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def called_names(node: ast.AST) -> set:
    return {call.func.id for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}


def attribute_calls(node: ast.AST, attribute: str) -> list:
    return [call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == attribute]


def first_text_argument(call: ast.Call) -> str | None:
    candidates = list(call.args)
    candidates += [keyword.value for keyword in call.keywords
                   if keyword.arg in {"label", "name"}]
    for candidate in candidates:
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            return candidate.value
    return None


def is_button_guard(node: ast.AST, label: str) -> bool:
    """Match `if st.button("<label>", ...):`."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Call):
        return False
    call = node.test
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "button"):
        return False
    text = first_text_argument(call)
    return bool(text) and label in text


def is_subheader(statement: ast.stmt) -> bool:
    value = getattr(statement, "value", None)
    return (isinstance(statement, ast.Expr) and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "subheader")


def statement_lists(tree: ast.Module):
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value:
                if all(isinstance(item, ast.stmt) for item in value):
                    yield value


def imports_by_module(tree: ast.Module, module: str) -> list:
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == module]


def imported_roots(tree: ast.Module) -> set:
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestPhaseDEntryPoint(unittest.TestCase):
    """The dashboard exposes one GID selector and one Proceed action."""

    def setUp(self):
        self.tree = parse_module(APP_PATH)
        self.person_search = function_node(self.tree, "show_cross_camera_matching")
        self.assertIsNotNone(self.person_search,
                             "Person Search workflow is missing from app.py.")

    def test_global_id_selector_lives_in_the_person_search_workflow(self):
        labels = [first_text_argument(call)
                  for call in attribute_calls(self.person_search, "selectbox")]
        self.assertIn("Global Person ID", labels,
                      "The Person Search tab must offer a Global Person ID selector.")
        self.assertIn("load_registry", called_names(self.person_search),
                      "The selector must read the saved registry, not a derived list.")

    def test_proceed_button_exists_in_the_person_search_workflow(self):
        labels = [first_text_argument(call) or ""
                  for call in attribute_calls(self.person_search, "button")]
        self.assertTrue(any("Proceed" in label for label in labels),
                        "The combined video needs a clear Proceed action.")

    def test_render_runs_only_after_proceed(self):
        calls = [call for call in ast.walk(self.tree)
                 if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                 and call.func.id == "render_combined_global_person_video"]
        self.assertEqual(len(calls), 1,
                         "The combined renderer must be called exactly once.")
        guards = [node for node in ast.walk(self.tree)
                  if is_button_guard(node, "Proceed")]
        self.assertEqual(len(guards), 1, "Expected exactly one Proceed guard.")
        guarded = {id(call) for call in ast.walk(guards[0])}
        self.assertIn(id(calls[0]), guarded,
                      "The combined video must not be generated by selecting a GID alone.")
        self.assertEqual(len(attribute_calls(guards[0], "spinner")), 1,
                         "The guarded render should show progress feedback.")

    def test_selected_video_is_displayed_from_the_current_manifest(self):
        self.assertIn("combined_video_status", called_names(self.tree),
                      "The saved combined video must be revalidated before display.")


class TestPhaseDImportWiring(unittest.TestCase):
    """New helpers are imported from the modules that define them."""

    def setUp(self):
        self.tree = parse_module(APP_PATH)
        self.renderer = parse_module(VIDEO_RESULTS_PATH)

    def test_combined_helpers_are_imported_once_from_video_results(self):
        imports = imports_by_module(self.tree, "src.reid.video_results")
        self.assertEqual(len(imports), 1,
                         "app.py should import the video helpers exactly once.")
        names = {alias.name for alias in imports[0].names}
        self.assertTrue({
            "render_all_camera_tracking_videos",
            "render_combined_global_person_video",
            "combined_video_status",
        } <= names)

    def test_registry_module_is_still_imported_exactly_once(self):
        imports = imports_by_module(self.tree, "src.reid.global_registry")
        self.assertEqual(len(imports), 1,
                         "Phase C keeps one global registry import block in app.py.")
        names = {alias.name for alias in imports[0].names}
        self.assertNotIn("identity_members_all", names,
                         "The app should ask the renderer for members, not the registry.")

    def test_identity_members_all_is_used_inside_the_combined_renderer(self):
        imports = imports_by_module(self.renderer, "src.reid.global_registry")
        self.assertEqual(len(imports), 1)
        names = {alias.name for alias in imports[0].names}
        self.assertIn("identity_members_all", names)
        combined = function_node(self.renderer, "render_combined_global_person_video")
        self.assertIn("identity_members_all", called_names(combined))


class TestPhaseDRendererScope(unittest.TestCase):
    """The combined renderer is additive and never writes an identity."""

    def setUp(self):
        self.renderer = parse_module(VIDEO_RESULTS_PATH)
        self.combined = function_node(
            self.renderer, "render_combined_global_person_video",
        )
        self.legacy = function_node(
            self.renderer, "render_all_camera_tracking_videos",
        )
        self.assertIsNotNone(self.combined)
        self.assertIsNotNone(self.legacy)

    def test_combined_renderer_never_writes_the_registry(self):
        called = called_names(self.combined)
        self.assertEqual(called & REGISTRY_WRITERS, set(),
                         "Combined rendering must not assign or reject identities.")

    def test_combined_renderer_reads_sources_and_csvs_directly(self):
        called = called_names(self.combined)
        self.assertIn("_camera_source", called)
        self.assertIn("plan_frame_boxes", called)
        self.assertIn("compose_tile", called)
        self.assertIn("compose_end_card", called)
        self.assertNotIn("render_selected_person_video", called,
                         "The combined renderer must not re-encode per-camera MP4s.")
        opener = function_node(self.renderer, "_camera_source")
        self.assertIsNotNone(opener)
        self.assertIn("load_track_index", called_names(opener),
                      "Tiles must be driven by the saved track rows, not by re-detection.")

    def test_per_camera_renderer_keeps_its_original_behaviour(self):
        called = called_names(self.legacy)
        self.assertIn("identity_members", called)
        self.assertNotIn("identity_members_all", called,
                         "Phase C/D must not change how per-camera videos pick a track.")
        self.assertIn("render_selected_person_video", called)
        for helper in ("compose_grid", "compose_tile", "render_combined_global_person_video"):
            self.assertNotIn(helper, called)

    def test_no_new_runtime_dependencies(self):
        renderer_roots = imported_roots(self.renderer)
        self.assertTrue({"cv2", "numpy"} <= renderer_roots)
        self.assertEqual(
            renderer_roots & {"imageio", "moviepy", "torch", "pandas",
                              "PIL", "streamlit"},
            set(), "The combined renderer must stay on cv2 and numpy.",
        )
        app_roots = imported_roots(parse_module(APP_PATH))
        self.assertEqual(
            app_roots & {"imageio", "moviepy", "torch", "pandas"},
            set(), "Phase D must not add a dashboard dependency.",
        )


class TestPhaseDPresentation(unittest.TestCase):
    """The new block follows the existing heading conventions."""

    def test_no_section_renders_two_consecutive_subheaders(self):
        tree = parse_module(APP_PATH)
        duplicates = [first.lineno for statements in statement_lists(tree)
                      for first, second in zip(statements, statements[1:])
                      if is_subheader(first) and is_subheader(second)]
        self.assertEqual(duplicates, [],
                         f"Consecutive subheaders duplicate a heading on lines {duplicates}.")

    def test_combined_block_uses_a_markdown_heading(self):
        tree = parse_module(APP_PATH)
        person_search = function_node(tree, "show_cross_camera_matching")
        headings = [first_text_argument(call) or ""
                    for call in attribute_calls(person_search, "markdown")]
        self.assertTrue(any(heading.startswith("#### Combined") for heading in headings),
                        "The combined block should keep the existing markdown style.")


if __name__ == "__main__":
    unittest.main()

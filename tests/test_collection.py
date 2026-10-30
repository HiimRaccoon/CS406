import json

import pytest

from src.pilot.collect_results import collect, export_reviewed_recipe


def test_no_fabricated_results_or_nonpilot_runs(tmp_path):
    assert collect(tmp_path) == []
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "run_summary.json").write_text(json.dumps({"purpose": "official"}))
    with pytest.raises(ValueError, match="Not a pilot"):
        collect(tmp_path)


def test_export_requires_explicit_complete_reviewed_values(tmp_path):
    source = tmp_path / "review.json"
    source.write_text(json.dumps({"lr_candidates": [1e-5]}))
    with pytest.raises(ValueError, match="missing keys"):
        export_reviewed_recipe(source, "training_recipe_v1", [], tmp_path / "out")
    assert not (tmp_path / "out").exists()

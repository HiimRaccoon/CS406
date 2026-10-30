import pytest

from src.pilot.dataset import PilotDataset, resolve_image
from src.preprocessing.geoguessr_v1 import Preprocessor


def test_exact_split_and_label_ids(manifest):
    assert dict(manifest.counts) == {"train": 34997, "val": 7497, "test": 7502}
    assert len(manifest.labels) == 124
    assert len({r.label_id for r in manifest.for_split("val")}) == 106
    assert len(manifest.fingerprints) == 4


def test_test_images_are_never_opened(manifest, tmp_path):
    with pytest.raises(ValueError, match="metadata only"):
        PilotDataset(manifest, tmp_path, "test", Preprocessor(), 11)
    with pytest.raises(ValueError, match="metadata only"):
        manifest.verify_images(tmp_path, ("test",))
    with pytest.raises(ValueError, match="escapes"):
        resolve_image(tmp_path, "../outside.jpg")


def test_tampered_artifact_rejected(tmp_path):
    from src.pilot.dataset import FixedManifest
    (tmp_path / "manifest.csv").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        FixedManifest(tmp_path)

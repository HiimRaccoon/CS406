import numpy as np
import pytest
import torch
from PIL import Image

from src.pilot.conformance import verify_sample
from src.preprocessing.geoguessr_v1 import Photometric, Preprocessor, half_up


@pytest.mark.parametrize("wh", [(1536, 662), (1030, 703)])
def test_frozen_shapes_fill_and_raw_crop_source(wh):
    # A spatial gradient detects accidental global-resize-then-crop.
    w, h = wh
    y, x = np.indices((h, w))
    image = Image.fromarray(np.stack((x % 256, y % 256, (x + y) % 256), -1).astype("uint8"))
    p = Preprocessor("e2")
    assert verify_sample(p, image)["local_source_raw"]
    raw, valid, layout = p.processed_raw(image, Photometric(1.05, .95))
    views = p(image, Photometric(1.05, .95))
    for i, (left, top, width, height) in enumerate(layout["local_crops_xywh"]):
        expected = p.normalize(p.resize(raw[:, top:top+height, left:left+width], (320, 320)))
        assert torch.equal(views["local"][i], expected)
    assert (p.normalize(raw)[:, ~valid] == 0).all()
    # Pixels outside predictable masks retain navigation/scene content.
    plain, valid, _ = p.processed_raw(image)
    expected = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    assert torch.equal(plain[:, valid], expected[:, :h-14, :][:, valid])


def test_no_implicit_layout_or_resolution_changes():
    with pytest.raises(ValueError, match="Unknown raw layout"):
        Preprocessor()(Image.new("RGB", (640, 480)))
    with pytest.raises(ValueError, match="frozen"):
        Preprocessor("e2", (512, 256))
    assert half_up(2.5) == 3
    assert Preprocessor("e1hr", (1344, 576))(Image.new("RGB", (1536, 662)))["global"].shape == (3, 576, 1344)

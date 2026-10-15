"""Quick checks for the static site and its exported CNN weights.

These are packaging checks, not a claim that the model is biologically accurate.
"""
import base64
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SHAPES = {
    "enc1.0.weight": (3, 3, 3, 8), "enc1.0.bias": (8,),
    "enc1.2.weight": (3, 3, 8, 8), "enc1.2.bias": (8,),
    "enc2.0.weight": (3, 3, 8, 16), "enc2.0.bias": (16,),
    "enc2.2.weight": (3, 3, 16, 16), "enc2.2.bias": (16,),
    "bottleneck.0.weight": (3, 3, 16, 24), "bottleneck.0.bias": (24,),
    "bottleneck.2.weight": (3, 3, 24, 24), "bottleneck.2.bias": (24,),
    "up2.0.weight": (3, 3, 40, 16), "up2.0.bias": (16,),
    "up2.2.weight": (3, 3, 16, 16), "up2.2.bias": (16,),
    "up1.0.weight": (3, 3, 24, 8), "up1.0.bias": (8,),
    "up1.2.weight": (3, 3, 8, 8), "up1.2.bias": (8,),
    "out.weight": (1, 1, 8, 1), "out.bias": (1,),
}

class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.sources.extend(value for key, value in attrs if key == "src" and value)


def test_page_loads_real_inference_code():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    parser = Scripts()
    parser.feed(page)
    sources = [src.split("?", 1)[0].removeprefix("./") for src in parser.sources]
    assert "inference.js" in sources
    assert all((ROOT / src).is_file() for src in sources)
    engine = (ROOT / "inference.js").read_text(encoding="utf-8")
    assert "model_weights.json" in engine
    assert "tf.sigmoid" in engine
    assert "findPeaks" in engine


def test_exported_cnn_weights_are_complete():
    model = json.loads((ROOT / "model_weights.json").read_text(encoding="utf-8"))
    assert model["format"] == "pytorch-conv2d-to-tfjs-raw"
    assert model["trained_epochs"] == 32
    assert model["threshold"] == 0.75
    assert set(model["weights"]) == set(EXPECTED_SHAPES)
    for name, dims in EXPECTED_SHAPES.items():
        entry = model["weights"][name]
        assert tuple(entry["shape"]) == dims, name
        raw = base64.b64decode(entry["b64"], validate=True)
        size = 1
        for dim in dims:
            size *= dim
        assert len(raw) == size * 4, name

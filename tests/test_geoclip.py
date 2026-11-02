"""GeoCLIP weights/presets with a stubbed CLIP backbone (no 1.7 GB download)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture
def stub_clip(monkeypatch):
    import torch.nn as nn

    import geoclip.model.image_encoder as ie

    class FakeCLIP(nn.Module):
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def get_image_features(self, pixel_values):
            # Emulates transformers>=5 returning an output object instead of a tensor.
            from transformers.modeling_outputs import BaseModelOutputWithPooling

            return BaseModelOutputWithPooling(pooler_output=torch.ones(pixel_values.shape[0], 768))

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros(1, 3, 224, 224)}

    monkeypatch.setattr(ie, "CLIPModel", FakeCLIP)
    monkeypatch.setattr(ie, "AutoProcessor", FakeProcessor)


@pytest.mark.parametrize(("preset", "gallery_size"), [("world", 100_000), ("ukraine_russia", 15_725)])
def test_presets_load_real_weights(stub_clip, plain_png, preset, gallery_size):
    from geoclip import GeoCLIP

    model = GeoCLIP(preset=preset)
    assert model.gps_gallery.shape == (gallery_size, 2)
    gps, probs = model.predict(str(plain_png), top_k=3)
    assert gps.shape == (3, 2) and probs.shape == (3,)
    assert float(probs.sum()) <= 1.0 + 1e-6
    assert bool((gps[:, 0].abs() <= 90).all() and (gps[:, 1].abs() <= 180).all())


def test_timestamp_spellings_and_errors(stub_clip):
    from geoclip import GeoCLIP

    assert GeoCLIP(timestamp="08-29-02:48").timestamp == "08-29-02:48"
    GeoCLIP(timestamp="08-29-02-48")
    with pytest.raises(ValueError):
        GeoCLIP(preset="mars")
    with pytest.raises(FileNotFoundError):
        GeoCLIP(timestamp="01-01-00-00")


def test_engine_wrapper(stub_clip, plain_png):
    from geoosint.engines import geoclip_engine

    points = geoclip_engine.predict(str(plain_png), top_k=2, preset="world")
    assert len(points) == 2 and points[0].label == "GeoCLIP #1"

"""`flide run` rows without a GPU: a row's request inputs reach the family as the server would pass them.

    python tests/test_offline_request.py
"""
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from flide.runtime.offline import _request


def _data_url() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 48), (200, 30, 30)).save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


QUESTIONS = {"red": {"type": "noul", "instructions": "The image is red"}}


def test_images_reach_the_request():
    url = _data_url()
    req = _request({"id": "a1", "state": "one photo", "images": [url], "questions": QUESTIONS})
    assert getattr(req, "images", None) == [url], "images were dropped from the row"


def test_other_inputs_reach_the_request():
    media = [{"type": "image", "path": "p.jpg"}]
    req = _request({"state": "s", "media": media, "videos": [["f0.jpg"]], "video": "v.mp4",
                    "assets": {"a": 1}, "questions": QUESTIONS})
    assert req.media == media
    assert req.videos == [["f0.jpg"]] and req.video == "v.mp4" and req.assets == {"a": 1}


def test_bookkeeping_keys_stay_out():
    req = _request({"id": "a1", "label": True, "state": "s", "questions": QUESTIONS})
    assert not hasattr(req, "id") and not hasattr(req, "label") and not hasattr(req, "images")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

# this script serves a web page for annotating images using the model from seg.py
# you should be fine without touching this file at all

from argparse import ArgumentParser
from flask import Flask, send_file, request
from waitress import serve
from PIL import Image
import numpy as np
from pathlib import Path
from io import BytesIO
import json
from threading import Lock
from seg import Segmentor

app = Flask(__name__, static_url_path="")

seg = Segmentor()

anno_lock = Lock()
mask_lock = Lock()


def get_img_path(img_id):
    return Path(app.config["DATA_DIR"]) / "imgs" / f"{img_id}.jpg"


def get_mask_path(img_id):
    return Path(app.config["DATA_DIR"]) / "masks" / f"{img_id}.npz"


def _get_anno_path(img_id):
    return Path(app.config["DATA_DIR"]) / "output" / f"{img_id}.json"


def load_masks(img_id):
    p = get_mask_path(img_id)
    if p.exists():
        return dict(np.load(p))
    return {}


def mask2img(mask):
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask] = 255
    return Image.fromarray(rgba)


def send_pil(pil_img, format="JPEG"):
    img_io = BytesIO()
    pil_img.save(img_io, format, quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


@app.get("/img/<img_id>")
def get_img(img_id):
    img_path = get_img_path(img_id)
    if not img_path.exists():
        return "Image does not exist", 404

    pil_img = Image.open(img_path)
    return send_pil(pil_img)


def _load_anno(img_id):
    anno_path = _get_anno_path(img_id)
    if anno_path.exists():
        with anno_lock:
            with anno_path.open() as f:
                return json.load(f)
    return {"img_id": img_id, "relations": [], "instances": []}


@app.get("/annotation/<img_id>")
def get_annotation(img_id):
    anno = _load_anno(img_id)
    instances = anno["instances"]
    # add clicks to the instances
    with mask_lock:
        masks = load_masks(img_id)
    for i, inst in enumerate(instances):
        clicks = masks[f"clicks_{inst['uid']}"]
        instances[i]["clicks"] = [
            {"isPositive": bool(c[2] == 1), "x": c[0], "y": c[1]} for c in clicks
        ]
    return instances


@app.post("/annotation/<img_id>")
def post_annotation(img_id):
    anno = _load_anno(img_id)
    instances_data = request.get_json()
    anno["instances"] = instances_data
    out_path = _get_anno_path(img_id)
    with anno_lock:
        with out_path.open("w") as f:
            json.dump(anno, f, indent=2)
    return "ok"


@app.post("/undo/<img_id>/<seg_uid>")
def post_undo(img_id, seg_uid):
    """Undo the last click and return the new annotation. The mask has to be requested separately"""
    # remove the last click and mask from the .npz file
    # if there are zero clicks now, remove the annotation from the JSON
    with mask_lock:
        masks = load_masks(img_id)
        click_id = f"clicks_{seg_uid}"
        if seg_uid in masks and click_id in masks:
            anno = _load_anno(img_id)
            removed_all = len(masks[seg_uid]) == 1
            if removed_all:
                masks.pop(seg_uid)
                masks.pop(click_id)
                # also update the annotation file
                anno["instances"] = [
                    inst for inst in anno["instances"] if inst["uid"] != seg_uid
                ]
                with anno_lock:
                    with _get_anno_path(img_id).open("w") as f:
                        json.dump(anno, f, indent=2)
            else:
                masks[click_id] = masks[click_id][:-1]
                masks[seg_uid] = masks[seg_uid][:-1]

            # save the new masks
            np.savez_compressed(get_mask_path(img_id), **masks)

            if removed_all:
                # no mask, send status code 204 (No Content)
                return "", 204
            else:
                return send_pil(mask2img(masks[seg_uid][-1]), format="PNG")
        else:
            return "invalid", 400


@app.post("/mask/<img_id>/<seg_uid>")
def post_mask(img_id, seg_uid):
    click = request.get_json()

    with mask_lock:
        masks = load_masks(img_id)

        click_id = f"clicks_{seg_uid}"
        logit_id = f"logits_{seg_uid}"
        click_data = np.array((click["x"], click["y"], click["isPositive"] * 1))
        if click_id in masks:
            # check if the click already exists
            if (masks[click_id] == click_data).all(1).any():
                # click exists already, ignore it
                print("duplicate click, ignoring")
                return send_pil(mask2img(masks[seg_uid][-1]), format="PNG")
            masks[click_id] = np.concatenate(
                (masks[click_id], click_data[None]), axis=0
            )
        else:
            masks[click_id] = click_data[None]

        prev_logits = None
        if seg_uid in masks:
            prev_logits = masks[logit_id]

        print("Running segmentation model")
        new_mask, new_logits = seg.segment(get_img_path(img_id), masks[click_id].copy(), prev_logits)

        if seg_uid in masks:
            masks[seg_uid] = np.concatenate((masks[seg_uid], new_mask[None]), axis=0)
            masks[logit_id] = np.concatenate((masks[logit_id], new_logits[None]), axis=0)
        else:
            masks[seg_uid] = new_mask[None]
            masks[logit_id] = new_logits[None]

        np.savez_compressed(get_mask_path(img_id), **masks)
    return send_pil(mask2img(new_mask), format="PNG")


@app.post("/prepare/<img_id>")
def prepare_embeddings(img_id):
    seg.prepare_embeddings(get_img_path(img_id))
    return "ready"


@app.get("/mask/<img_id>/<seg_uid>")
def get_mask(img_id, seg_uid):
    with mask_lock:
        masks = load_masks(img_id)
    if seg_uid in masks:
        last_mask = masks[seg_uid][-1]
        return send_pil(mask2img(last_mask), format="PNG")
    return "Mask not found", 404


def set_data_dir(data_root):
    assert (
        data_root / "imgs"
    ).is_dir(), f"Put your image files into {data_root/'imgs'}"
    (data_root / "output").mkdir(exist_ok=True, parents=True)
    app.config.from_mapping(DATA_DIR=data_root)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--data", default=None, type=str)
    args = parser.parse_args()

    if args.data:
        set_data_dir(Path(args.data))
    else:
        set_data_dir(Path("data"))

    print(
        f"Open this link in your web browser: http://localhost:{args.port}/index.html"
    )
    serve(app, host="localhost", port=args.port)

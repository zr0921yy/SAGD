# -*- coding: utf-8 -*-
"""Merge AerialGaze-1.0 gaze stimulus tiles into the detection COCO train set.

The eye-tracking dataset ships its stimulus images (images/<class>/<stem>.jpg)
and DOTA-format per-tile labels (annfiles/<class>/<stem>.txt). The stimulus
tiles follow the official split grid (patch 1024 / gap 200, mmrotate naming
<src>__1024__<x>___<y>) but are usually NOT part of the standard DOTA train
tiles — merge them so gaze density maps can be matched by stem at training time.

Steps: copy images into --stim-dir, then append image + annotation entries
(bounding box of the polygon, no clipping) to the COCO json. Idempotent:
already-merged tiles are skipped.
"""
import argparse
import json
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from convert_dota15_to_coco import DOTA15_CATEGORIES  # noqa: E402


def parse_tile_txt(txt_path):
    objects = []
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 9 or parts[0] in ("imagesource", "gsd"):
            continue
        xs = [float(v) for v in parts[0:8:2]]
        ys = [float(v) for v in parts[1:8:2]]
        x0, y0 = min(xs), min(ys)
        objects.append((x0, y0, max(xs) - x0, max(ys) - y0, parts[8]))
    return objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze-root", required=True,
                    help="AerialGaze-1.0 root (images/ + annfiles/)")
    ap.add_argument("--coco", required=True,
                    help="train COCO json to update (in place, .bak backup kept)")
    ap.add_argument("--stim-dir", required=True,
                    help="folder to store copied stimulus images (e.g. data/DOTA1.5_sliced/stimuli)")
    ap.add_argument("--file-prefix", default="stimuli",
                    help="prefix for COCO file_name (default stimuli)")
    args = ap.parse_args()

    gaze_root = Path(args.gaze_root)
    stim_root = Path(args.stim_dir)
    coco = json.load(open(args.coco, encoding="utf-8"))
    cat_id = {c["name"]: c["id"] for c in coco["categories"]}
    existing = {im["file_name"] for im in coco["images"]}
    img_id = max((im["id"] for im in coco["images"]), default=40000000) + 1
    ann_id = max((a["id"] for a in coco["annotations"]), default=50000000) + 1

    n_img = n_ann = n_skip = 0
    for txt in sorted(gaze_root.glob("annfiles/*/*.txt")):
        cls = txt.parent.name
        stem = txt.stem
        fn = f"{args.file_prefix}/{cls}/{stem}.jpg"
        if fn in existing:
            n_skip += 1
            continue
        src = gaze_root / "images" / cls / (stem + ".jpg")
        dst = stim_root / cls / (stem + ".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"[warn] 刺激图缺失, 跳过: {src}")
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
        coco["images"].append({"id": img_id, "file_name": fn,
                               "width": 1024, "height": 1024})
        for x0, y0, w, h, name in parse_tile_txt(txt):
            if name not in cat_id:
                print(f"[warn] 未知类别 {name} @ {stem}")
                continue
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id[name],
                "bbox": [round(x0, 1), round(y0, 1), round(w, 1), round(h, 1)],
                "area": round(w * h, 1), "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1
            n_ann += 1
        img_id += 1
        n_img += 1

    shutil.copy2(args.coco, str(args.coco) + ".bak")
    json.dump(coco, open(args.coco, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"完成: 合并 {n_img} 张刺激图 / {n_ann} 条标注 (已存在跳过 {n_skip})")
    print(f"备份: {args.coco}.bak")


if __name__ == "__main__":
    main()

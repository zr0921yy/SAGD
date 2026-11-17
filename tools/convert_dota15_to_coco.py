# -*- coding: utf-8 -*-
"""Per-tile DOTA labelTxt -> COCO json (DOTA-1.5, 16 classes, category_id 1-16).

Input:  --img-dir  <split>/images/*.jpg   (official mmrotate/DOTA-devkit split naming)
        --label-dir <split>/labelTxt/*.txt  (per-tile DOTA txt, coords in tile space,
                                             optional 'imagesource'/'gsd' header lines)
Output: COCO json with images/annotations/categories, compatible with datasets/coco.py.

NOTE: run official splitting first (mmrotate tools/data/dota/split/img_split.py,
patch 1024 / gap 200), then this script.
"""
import argparse
import json
from pathlib import Path

DOTA15_CATEGORIES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
    "harbor", "swimming-pool", "helicopter", "container-crane",
]


def parse_tile_txt(txt_path):
    """Parse one per-tile DOTA txt -> [(bbox, name), ...]; skips header lines."""
    objects = []
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 9 or parts[0] in ("imagesource", "gsd"):
            continue
        xs = [float(v) for v in parts[0:8:2]]
        ys = [float(v) for v in parts[1:8:2]]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0, max(ys) - y0
        objects.append((x0, y0, w, h, parts[8]))
    return objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--file-prefix", default="train/images",
                    help="prefix stored in COCO file_name (default train/images)")
    args = ap.parse_args()

    cat_id = {n: i + 1 for i, n in enumerate(DOTA15_CATEGORIES)}
    img_dir, label_dir = Path(args.img_dir), Path(args.label_dir)

    images, annotations = [], []
    img_id, ann_id = 1, 1
    tiles = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".png"))
    for p in tiles:
        txt = label_dir / (p.stem + ".txt")
        if not txt.exists():
            print(f"[warn] 无标注, 跳过: {p.name}")
            continue
        images.append({"id": img_id,
                       "file_name": f"{args.file_prefix}/{p.name}",
                       "width": 1024, "height": 1024})
        for x0, y0, w, h, name in parse_tile_txt(txt):
            if name not in cat_id:
                print(f"[warn] 未知类别 {name} @ {p.name}")
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id[name],
                "bbox": [round(x0, 1), round(y0, 1), round(w, 1), round(h, 1)],
                "area": round(w * h, 1), "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1
        img_id += 1

    out = {"images": images, "annotations": annotations,
           "categories": [{"id": i + 1, "name": n, "supercategory": "none"}
                          for i, n in enumerate(DOTA15_CATEGORIES)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"完成: {len(images)} 张图, {len(annotations)} 条标注 -> {args.out}")


if __name__ == "__main__":
    main()

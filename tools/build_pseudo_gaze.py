# -*- coding: utf-8 -*-
"""统计迁移伪注视密度图：真人注视 → 全训练集的蒸馏目标（SAGD 臂2）。

一次运行内完成两个阶段：
  fit   在「有人眼密度图」的 tile 上，统计每个 GT 框内部的注视空间分布
        （框内相对坐标的 K×K 概率网格）以及该框占整图注视质量的份额 share，
        按 类别 × 面积档 聚合（--area-bins 控制档数）。
  apply 对标注文件里的每张训练图，按每个 GT 框的类别/面积档查表，按预测
        share 分配伪注视点预算（--points），从对应网格采样后做 σ 高斯溅射，
        写出与 datasets/coco.py 完全一致的格式：
          <out>/<task_dir>/<class>/<stem>.npz   float16 (res, res)
          <out>/index.json

查表回退（保证所有类别都有目标）：
  (class, area-bin) → (class, 全部面积) → (全部类别, area-bin) → (全部类别, 全部面积)

与真人密度图（tools/build_gaze_density.py）的关系：
  - 面积档边界取标注文件全部框 sqrt(w·h) 的分位数；
  - 只在框内注视质量 > 0 的框上拟合，避免空白框稀释分布；
  - 输出仅用于训练，加载端仍逐图 max 归一化，因此密度绝对值无意义；
  - 生成结果覆盖所有有标注的训练 tile（不依赖任何眼动原始数据）。

示例：
  python tools/build_pseudo_gaze.py \
      --coco data/DOTA1.5_sliced/annotations/instances_train2017.json \
      --gaze-dir data/gaze_density \
      --out data/pseudo_density
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_RESOLUTION = 1024
DEFAULT_SIGMA = 16.0
ALL = "*"    # 全类别回退键


def parse_args():
    ap = argparse.ArgumentParser(
        description="Statistical-transfer pseudo-gaze density maps (box statistics from real gaze).")
    ap.add_argument("--coco", required=True,
                    help="检测标注 COCO json（提供框、类别与图名）")
    ap.add_argument("--gaze-dir", required=True,
                    help="真人密度图目录（tools/build_gaze_density.py 的输出，用于拟合统计）")
    ap.add_argument("--out", default="data/pseudo_density",
                    help="输出根目录（默认 data/pseudo_density）")
    ap.add_argument("--task-dir", default="search",
                    help="输出子目录名，与 datasets/coco.py 的 gaze_task 对应（默认 search）")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="只迁移这些类别（默认全部；建议与真人统计的类别一致）")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help="输出密度图边长（默认 1024）")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                    help="伪注视高斯核 σ，单位像素（默认 16，与真人密度图一致）")
    ap.add_argument("--grid", type=int, default=8,
                    help="框内相对坐标网格边长 K（默认 8）")
    ap.add_argument("--area-bins", type=int, default=3,
                    help="框面积档数量（默认 3；1 表示不分档）")
    ap.add_argument("--points", type=int, default=32,
                    help="每张图的伪注视点预算（默认 32，按各框 share 分配）")
    ap.add_argument("--seed", type=int, default=42,
                    help="采样随机种子（默认 42）")
    ap.add_argument("--skip-gaze-covered", action="store_true",
                    help="跳过已有人眼密度图的 tile（默认不跳过，便于做完整替代对照）")
    return ap.parse_args()


def load_coco(coco_path):
    """读取 COCO：图列表 + 每图标注框（类别名, xywh）。"""
    data = json.load(open(coco_path, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = [{
        "id": im["id"],
        "stem": Path(im["file_name"]).stem,
        "width": im["width"],
        "height": im["height"],
    } for im in data["images"]]
    boxes = defaultdict(list)
    for ann in data["annotations"]:
        boxes[ann["image_id"]].append({
            "cls": cats.get(ann["category_id"], str(ann["category_id"])),
            "bbox": [float(v) for v in ann["bbox"]],      # xywh
        })
    return images, boxes


def build_gaze_index(gaze_dir, task_dir):
    """递归索引真人密度图，stem → npz 路径（排序保证跨类重复时选择确定）。"""
    base = Path(gaze_dir)
    task_base = base / task_dir
    if task_base.is_dir():
        base = task_base
    idx = {}
    for f in sorted(base.rglob("*.npz")):
        idx.setdefault(f.stem, f)
    return idx


def build_area_edges(boxes, n_bins):
    """面积档边界：全部标注框 sqrt(w·h) 的分位数。"""
    sides = [np.sqrt(max(b["bbox"][2], 0.0) * max(b["bbox"][3], 0.0))
             for bs in boxes.values() for b in bs]
    sides = np.asarray([v for v in sides if v >= 1.0], dtype=np.float64)
    if n_bins <= 1 or sides.size == 0:
        return np.zeros(0, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(sides, qs)


def area_bin_of(bbox, edges):
    side = np.sqrt(max(bbox[2], 0.0) * max(bbox[3], 0.0))
    return int(np.searchsorted(edges, side, side="right"))


def block_sum(arr, k):
    """把任意尺寸数组按均匀边界分块求和成 k×k（小框也能处理）。"""
    out = np.zeros((k, k), dtype=np.float64)
    ys = np.linspace(0, arr.shape[0], k + 1).astype(int)
    xs = np.linspace(0, arr.shape[1], k + 1).astype(int)
    for i in range(k):
        y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
        for j in range(k):
            x0, x1 = xs[j], max(xs[j + 1], xs[j] + 1)
            out[i, j] = arr[y0:y1, x0:x1].sum()
    return out


def clip_box(bbox, scale_x, scale_y, width, height):
    """标注框 → 密度图网格，返回整数裁剪区 (x0, y0, x1, y1) 或 None。"""
    x, y, w, h = bbox
    x0 = max(0, int(np.floor(x * scale_x)))
    y0 = max(0, int(np.floor(y * scale_y)))
    x1 = min(width, int(np.ceil((x + w) * scale_x)))
    y1 = min(height, int(np.ceil((y + h) * scale_y)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x0, y0, x1, y1


def fit_stats(images, boxes, gaze_index, edges, k, keep_classes):
    """在真人覆盖 tile 上拟合 类别×面积档 的框内分布与注视份额。"""
    acc = defaultdict(lambda: {"cell": np.zeros((k, k), np.float64),
                               "share": 0.0, "n": 0})
    n_tiles = n_boxes = 0
    for im in images:
        p = gaze_index.get(im["stem"])
        if p is None:
            continue
        density = np.load(p)["density"].astype(np.float32)
        total = float(density.sum())
        if total <= 0:
            continue
        height, width = density.shape
        scale_x, scale_y = width / im["width"], height / im["height"]
        used = False
        for box in boxes.get(im["id"], []):
            if keep_classes and box["cls"] not in keep_classes:
                continue
            region = clip_box(box["bbox"], scale_x, scale_y, width, height)
            if region is None:
                continue
            x0, y0, x1, y1 = region
            crop = density[y0:y1, x0:x1]
            mass = float(crop.sum())
            if mass <= 0:      # 该框内没有注视，跳过以免稀释分布
                continue
            cell = block_sum(crop, k)
            cell = cell / cell.sum()
            share = mass / total
            b = area_bin_of(box["bbox"], edges)
            for key in ((box["cls"], b), (box["cls"], -1), (ALL, b), (ALL, -1)):
                entry = acc[key]
                entry["cell"] += cell * share       # 注视多的框对形状贡献更大
                entry["share"] += share
                entry["n"] += 1
            n_boxes += 1
            used = True
        if used:
            n_tiles += 1
    return acc, n_tiles, n_boxes


def lookup(acc, cls, b):
    """按回退链取 (框内概率网格, 平均 share, 命中的键)。"""
    for key in ((cls, b), (cls, -1), (ALL, b), (ALL, -1)):
        entry = acc.get(key)
        if entry and entry["n"] > 0 and entry["share"] > 0 and entry["cell"].sum() > 0:
            prob = entry["cell"] / entry["cell"].sum()
            return prob, entry["share"] / entry["n"], key
    return None, 0.0, None


def synthesize(images, boxes, gaze_index, acc, edges, args, out_root):
    """按拟合统计为每张有标注的训练图生成伪注视密度图。"""
    rng = np.random.default_rng(args.seed)
    res, k = args.resolution, args.grid
    n_tiles = n_boxes = n_points = 0
    fallback = Counter()
    for im in sorted(images, key=lambda x: x["stem"]):
        if args.skip_gaze_covered and im["stem"] in gaze_index:
            continue
        anns = boxes.get(im["id"], [])
        if not anns:
            continue
        entries = []
        for box in anns:
            if args.classes and box["cls"] not in args.classes:
                continue
            prob, share, key = lookup(acc, box["cls"], area_bin_of(box["bbox"], edges))
            if prob is None:
                continue
            entries.append((box, prob, share, key))
        if not entries:
            continue
        shares = np.asarray([max(e[2], 0.0) for e in entries], dtype=np.float64)
        if shares.sum() <= 0:
            shares = np.ones(len(entries), dtype=np.float64)
        shares = shares / shares.sum()

        scale_x, scale_y = res / im["width"], res / im["height"]
        points = []
        for (box, prob, _, key), share in zip(entries, shares):
            fallback[key] += 1
            n = max(1, int(round(args.points * share)))
            cells = rng.choice(prob.size, size=n, p=prob.ravel() / prob.ravel().sum())
            cy, cx = np.divmod(cells, k)
            jitter_x, jitter_y = rng.random(n), rng.random(n)
            x, y, w, h = box["bbox"]
            points.append(np.stack([
                (x + (cx + jitter_x) / k * w) * scale_x,
                (y + (cy + jitter_y) / k * h) * scale_y,
            ], axis=1))
            n_boxes += 1
        points = np.concatenate(points, axis=0)
        keep = ((points[:, 0] >= 0) & (points[:, 0] < res)
                & (points[:, 1] >= 0) & (points[:, 1] < res))
        points = points[keep]
        if points.size == 0:
            continue

        canvas = np.zeros((res, res), dtype=np.float32)
        np.add.at(canvas, (points[:, 1].astype(int), points[:, 0].astype(int)), 1.0)
        density = gaussian_filter(canvas, sigma=args.sigma, mode="constant", truncate=3.0)
        if density.max() <= 0:
            continue

        primary = max(entries, key=lambda e: e[0]["bbox"][2] * e[0]["bbox"][3])[0]["cls"]
        (out_root / primary).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_root / primary / f"{im['stem']}.npz",
                            density=density.astype(np.float16))
        n_tiles += 1
        n_points += int(points.shape[0])
    return n_tiles, n_boxes, n_points, fallback


def main():
    args = parse_args()
    keep_classes = set(args.classes) if args.classes else None
    if args.grid < 1 or args.points < 1:
        raise SystemExit("--grid/--points 必须 >= 1")

    images, boxes = load_coco(args.coco)
    gaze_index = build_gaze_index(args.gaze_dir, args.task_dir)
    if not gaze_index:
        raise SystemExit(f"--gaze-dir 中没有 npz：{args.gaze_dir}")
    edges = build_area_edges(boxes, args.area_bins)

    acc, fit_tiles, fit_boxes = fit_stats(
        images, boxes, gaze_index, edges, args.grid, keep_classes)
    if not acc:
        raise SystemExit("拟合失败：--coco 与 --gaze-dir 没有可匹配的『图 + 标注 + 密度图』")

    out_root = Path(args.out) / args.task_dir
    out_root.mkdir(parents=True, exist_ok=True)
    n_tiles, n_boxes, n_points, fallback = synthesize(
        images, boxes, gaze_index, acc, edges, args, out_root)

    index = {
        "method": "statistical transfer (class x area-bin)",
        "resolution": args.resolution,
        "sigma": args.sigma,
        "grid": args.grid,
        "area_bins": args.area_bins,
        "area_edges": [float(v) for v in edges],
        "points_per_tile": args.points,
        "seed": args.seed,
        "skip_gaze_covered": bool(args.skip_gaze_covered),
        "classes": sorted(keep_classes) if keep_classes else None,
        "fit": {"n_tiles": fit_tiles, "n_boxes": fit_boxes},
        "apply": {"n_tiles": n_tiles, "n_boxes": n_boxes, "n_points": n_points},
        "fallback_usage": {f"{key[0]}/{key[1]}": c for key, c in sorted(fallback.items())},
        "support": {f"{key[0]}/{key[1]}": acc[key]["n"] for key in sorted(acc)},
    }
    json.dump(index, open(Path(args.out) / "index.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"拟合: {fit_tiles} 张真人覆盖 tile / {fit_boxes} 个框")
    print(f"迁移: {n_tiles} 张伪密度图 / {n_boxes} 个框 / {n_points} 个伪注视点")
    print(f"输出: {out_root.resolve()}")


if __name__ == "__main__":
    main()

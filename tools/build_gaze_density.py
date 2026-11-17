# -*- coding: utf-8 -*-
"""AerialGaze-1.0 公开眼动数据 → 逐 tile 人眼注视密度图（SAGD 蒸馏目标构建）。

输入：公开数据集的逐试次眼动 JSON（注视坐标已与 1024×1024 刺激图尺度对齐）。
      工具递归扫描 --src 下全部 *.json，按 JSON 内 meta 自描述字段分组
      （meta.participant / meta.class / meta.image），不依赖目录层级。

输出：与 datasets/coco.py 加载器约定一致——
      <out>/<task_dir>/<class>/<stem>.npz   float16 密度图 resolution×resolution
                                            （未归一化和；加载端会 max 归一）
      <out>/index.json                      构建参数 + 覆盖清单 + 校验报告

坐标约定（--coord-space）：
  pixel  （默认）像素坐标，直接索引 [0, resolution) 图像网格；
         超出范围的注视点丢弃并计数（dropped_out_of_range），不做边缘 clamp
  norm01 归一化坐标 [0,1]，乘以 resolution 转像素
  auto   依 meta.coord_space 判别（含 "norm" → norm01，否则 pixel）；
         meta 缺失时按数值启发式：单文件内全部坐标 ≤ 1 视为 norm01

融合协议（与论文训练协议一致）：
  - σ=16px 高斯溅射（约注视空间不确定度），核叠加、出图截断
  - 时长加权：权重 = max(dur, 0.05)（--no-duration-weight 关闭）
  - 多被试融合逐被试先 max 归一再相加（"一人一票"，防止注视数/时长差异主导）
    （--no-per-subject-norm 关闭）

校验（可选，推荐都开）：
  --images <刺激图目录>    类别子目录下的刺激图（<class>/<stem>.*）；
                           每个 npz stem 应能对上一张刺激图
  --coco-ann <COCO json>   与检测标注图名逐 stem 比对；对不上的列入
                           index.json 的 orphan_stems（不阻断、密度图照常
                           写出，交由使用者裁量——蒸馏目标靠 stem 匹配图片，
                           orphan 意味着该密度图不会被训练加载）

示例：
  python tools/build_gaze_density.py --src <AerialGaze-1.0 根目录> \
      --out data/gaze_density \
      --images <刺激图目录> \
      --coco-ann data/DOTA1.5_sliced/annotations/instances_train2017.json \
      --classes bridge court "ground track field" harborship plane roundabout \
                "storage tank" "swimming pool" vehicle
"""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_SIGMA = 16.0
DEFAULT_RESOLUTION = 1024
MIN_DUR = 0.05  # 秒；时长权重下限，防止极短注视被完全忽略


def splat(resolution, sigma, points, weights):
    """逐注视高斯溅射（直接核叠加，点数少时足够快）。"""
    m = np.zeros((resolution, resolution), np.float32)
    rr = int(3 * sigma)
    g1d = np.exp(-0.5 * (np.arange(-rr, rr + 1) / sigma) ** 2)
    for (x, y), w in zip(points, weights):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - rr), min(resolution, xi + rr + 1)
        y0, y1 = max(0, yi - rr), min(resolution, yi + rr + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        kx = g1d[x0 - (xi - rr): x1 - (xi - rr)]
        ky = g1d[y0 - (yi - rr): y1 - (yi - rr)]
        m[y0:y1, x0:x1] += w * np.outer(ky, kx)
    return m


def resolve_space(meta, fallback):
    """auto 模式下判定单文件坐标空间：meta.coord_space 优先，数值启发式兜底。"""
    if not fallback:  # 显式指定，无需判别
        return fallback
    cs = str(meta.get("coord_space", "")).lower()
    if "norm" in cs:
        return "norm01"
    if cs in ("pixel", "px", "image_pixel", "tile_pixel"):
        return "pixel"
    return fallback  # meta 缺失/不认识 → 由调用方的启发式决定


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build per-tile human gaze density maps from the AerialGaze-1.0 JSONs.")
    ap.add_argument("--src", required=True,
                    help="公开眼动数据集根目录（递归扫描 *.json）")
    ap.add_argument("--out", default="data/gaze_density",
                    help="输出目录（默认 data/gaze_density）")
    ap.add_argument("--task-dir", default="search",
                    help="输出子目录名，与 datasets/coco.py 的 gaze_task 默认值对应（默认 search）")
    ap.add_argument("--coord-space", choices=["pixel", "norm01", "auto"], default="pixel",
                    help="公开 JSON 的注视坐标表示（默认 pixel；auto 依 meta/数值判别）")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help="密度图边长（默认 1024，应与刺激图一致）")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                    help="高斯溅射 σ，单位像素（默认 16）")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="只处理这些类别（默认全部；目标搜索任务见示例）")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="只融合这些被试（meta.participant，默认全部）")
    ap.add_argument("--meta-task", default=None,
                    help="只处理 meta.task 等于该值的 JSON（字符串比对，默认不过滤）")
    ap.add_argument("--images", default=None,
                    help="刺激图目录（<class>/<stem>.*，用于 stem 校验）")
    ap.add_argument("--coco-ann", default=None,
                    help="检测 COCO 标注 json（用于 stem 孤儿校验，可多次外的逗号分隔）")
    ap.add_argument("--per-subject-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="逐被试 max 归一后再相加（默认开，即论文的一人一票协议）")
    ap.add_argument("--duration-weight", action=argparse.BooleanOptionalAction, default=True,
                    help="注视时长加权（默认开）")
    ap.add_argument("--respect-include", action=argparse.BooleanOptionalAction, default=True,
                    help="跳过 meta.include == 0 的试次（默认开）")
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"--src 不存在或不是目录: {src}")
    out = Path(args.out) / args.task_dir
    out.mkdir(parents=True, exist_ok=True)

    # 校验用 stem 集合
    img_stems = None
    if args.images:
        img_root = Path(args.images)
        img_stems = {p.parent.name: set() for p in img_root.glob("*/*") if p.is_file()}
        for p in img_root.glob("*/*"):
            if p.is_file():
                img_stems[p.parent.name].add(p.stem)
    coco_stems = set()
    if args.coco_ann:
        for ann in args.coco_ann.split(","):
            ann = ann.strip()
            data = json.load(open(ann, encoding="utf-8"))
            coco_stems |= {Path(im["file_name"]).stem for im in data["images"]}

    # ── 扫描 + 解析 ─────────────────────────────────────────────
    # trials[(cls, stem)][participant] = (points, weights)
    trials = {}
    stats = {"n_json": 0, "n_bad": 0, "n_skipped_include": 0, "n_skipped_filter": 0,
             "n_fix": 0, "n_dropped_oob": 0, "n_dup_trial": 0}
    for p in sorted(src.rglob("*.json")):
        stats["n_json"] += 1
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            stats["n_bad"] += 1
            continue
        meta = d.get("meta") or {}
        if args.respect_include and meta.get("include") == 0:
            stats["n_skipped_include"] += 1
            continue
        if args.meta_task is not None and str(meta.get("task")) != args.meta_task:
            stats["n_skipped_filter"] += 1
            continue
        cls = str(meta.get("class") or p.parent.name)
        stem = str(meta.get("image") or p.stem)
        pid = str(meta.get("participant") or p.parent.parent.name or "unknown")
        if args.classes and cls not in args.classes:
            stats["n_skipped_filter"] += 1
            continue
        if args.subjects and pid not in args.subjects:
            continue
        fx = d.get("fixations") or []
        if not fx:
            continue
        # 坐标空间判定
        space = args.coord_space
        if space == "auto":
            space = resolve_space(meta, "pixel")
            if space == "pixel" and meta.get("coord_space") is None:
                coords = [(f.get("x", 0), f.get("y", 0)) for f in fx]
                if all(abs(x) <= 1 and abs(y) <= 1 for x, y in coords):
                    space = "norm01"
        pts, wts = [], []
        for f in fx:
            x, y = float(f["x"]), float(f["y"])
            if space == "norm01":
                x, y = x * args.resolution, y * args.resolution
            if not (0 <= x < args.resolution and 0 <= y < args.resolution):
                stats["n_dropped_oob"] += 1
                continue
            pts.append((x, y))
            wts.append(max(float(f.get("dur", MIN_DUR)), MIN_DUR)
                       if args.duration_weight else 1.0)
        if not pts:
            continue
        key = (cls, stem)
        trials.setdefault(key, {})
        if pid in trials[key]:
            stats["n_dup_trial"] += 1
            if len(pts) <= len(trials[key][pid][0]):
                continue  # 已有更完整的同名试次
        trials[key][pid] = (pts, wts)
        stats["n_fix"] += len(pts)

    if stats["n_dup_trial"]:
        print(f"[warn] 同一被试同一 tile 出现多个 JSON，取注视数最多者（{stats['n_dup_trial']} 次）")

    # ── 逐 tile 融合 + 写出 ────────────────────────────────────
    index = {
        "sigma": args.sigma, "resolution": args.resolution,
        "coord_space": args.coord_space, "per_subject_norm": args.per_subject_norm,
        "duration_weight": args.duration_weight, "task_dir": args.task_dir,
        "n_tiles": 0, "n_fixations": stats["n_fix"], "items": {},
        "orphan_stems": [], "images_missing": [],
    }
    per_class = {}
    for (cls, stem), by_pid in sorted(trials.items()):
        density = np.zeros((args.resolution, args.resolution), np.float32)
        sources = []
        for pid, (pts, wts) in sorted(by_pid.items()):
            m = splat(args.resolution, args.sigma, pts, wts)
            if args.per_subject_norm and m.max() > 0:
                m = m / m.max()
            density += m
            sources.append({"participant": pid, "n_fix": len(pts)})
        if density.max() <= 0:
            continue
        (out / cls).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / cls / f"{stem}.npz",
                            density=density.astype(np.float16))
        index["items"][f"{cls}/{stem}"] = sources
        index["n_tiles"] += 1
        per_class[cls] = per_class.get(cls, 0) + 1
        # stem 校验
        if coco_stems and stem not in coco_stems:
            index["orphan_stems"].append(f"{cls}/{stem}")
        if img_stems is not None and cls not in img_stems:
            index["images_missing"].append(f"{cls}/{stem}")
        elif img_stems is not None and stem not in img_stems.get(cls, set()):
            index["images_missing"].append(f"{cls}/{stem}")

    index["per_class"] = per_class
    json.dump(index, open(Path(args.out) / "index.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ── 摘要 ───────────────────────────────────────────────────
    for cls in sorted(per_class):
        print(f"{cls:20s} {per_class[cls]:5d} tiles")
    print(f"总计 {index['n_tiles']} 张密度图 / {stats['n_fix']} 个注视"
          f"（解析失败 {stats['n_bad']}，include=0 跳过 {stats['n_skipped_include']}，"
          f"过滤跳过 {stats['n_skipped_filter']}，出界丢弃 {stats['n_dropped_oob']}）")
    if coco_stems:
        print(f"stem 校验：命中检测标注 {index['n_tiles'] - len(index['orphan_stems'])}"
              f"/{index['n_tiles']}，orphan {len(index['orphan_stems'])}"
              f"（明细见 index.json orphan_stems）")
    if img_stems is not None:
        print(f"刺激图校验：未对上刺激图的密度图 {len(index['images_missing'])} 张")
    print(f"输出: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

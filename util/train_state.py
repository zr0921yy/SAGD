# -*- coding: utf-8 -*-
"""RS-Gaze 半 epoch 存档/续训状态管理。

背景：checkpoint 只在 epoch 边界存在（accelerate automatic_checkpoint_naming，
目录号 = save 计数 = 已完成 epoch 数），一次驱动级挂起平均损失半个 epoch
（~35-55 min）。本模块把存档粒度降到 ~25 min（MID_EPOCH_SAVE_ITERS，默认
1500 iter，env 可调，0 = 关闭）：

- 训练中：engine 每迭代回调 saver(step)；到频点时 accelerator.save_state()
  落一个新 checkpoint_* 目录（与 epoch 边界同一条 accelerate 命名序列，由
  total_limit 统一轮换），随后写 <output_dir>/train_state.json 作为"提交
  标记"（kind / epoch / step / dir）。先落盘后写标记：挂起发生在两步之间
  时，标记仍指向旧的完整档，恢复语义自然安全（半写目录永远不会被引用）。
- 恢复时：按 train_state.json 推导 —— kind=mid → 本 epoch 第 step 批续跑
  （skip_batches=step）；kind=epoch_end → 下一 epoch。无标记（历史 run dir
  或标记丢失）回退"目录号+1"的历史语义，完全兼容既有目录。
- 批顺序语义：续跑 epoch 的批顺序与中断前不同（恢复点在 epoch 中段，采样
  器按恢复后的 RNG 重排）——对 SGD 无影响，等价于 epoch 中段重排；跳批的
  目的是省掉已训过的前向/反向时间，不是复现数据流。
- 目录号语义变化：启用后 checkpoint_* 目录号不再等于已完成 epoch 数（中间
  插入 mid 存档），epoch 推导一律走 train_state.json，目录号仅用于命名与
  排序。

单进程单卡约定（与训练现状一致）。所有落盘动作 try/except 包裹：存档失败
只记日志不中断训练——漏一次存档可接受，训练崩不可接受。
"""
import json
import logging
import os
import re
import shutil

logger = logging.getLogger("train_state")

STATE_FILE = "train_state.json"
# 默认 1500 iter ≈ 25 min（0.6 s/iter）；一次挂起平均损失半频点 ~12 min
MID_EPOCH_SAVE_ITERS = int(os.environ.get("MID_EPOCH_SAVE_ITERS", "1500"))


def _newest_checkpoint(output_dir):
    """返回（最新 checkpoint 目录路径, 目录号）；没有则 (None, -1)。"""
    ckpt_root = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None, -1
    best_path, best_num = None, -1
    for name in os.listdir(ckpt_root):
        m = re.fullmatch(r"checkpoint_(\d+)", name)
        if m and int(m.group(1)) > best_num:
            best_num = int(m.group(1))
            best_path = os.path.join(ckpt_root, name)
    return best_path, best_num


def _read_state(output_dir):
    """读 train_state.json；标记缺失或其目录已不存在 → None（不信任半标记）。"""
    path = os.path.join(output_dir, STATE_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if st.get("dir") and os.path.isdir(os.path.join(output_dir, "checkpoints", st["dir"])):
            return st
    except (OSError, ValueError, KeyError):
        pass
    return None


def _write_state(output_dir, st):
    path = os.path.join(output_dir, STATE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)  # 原子替换，防半写标记


def _remove_stale_mid_dir(output_dir, old_st, keep_dir_name):
    """旧标记指向的 mid 目录在滚动后删除（epoch 边界档永不在此路径上被删）。"""
    if old_st and old_st.get("kind") == "mid" and old_st.get("dir") not in (None, keep_dir_name):
        stale = os.path.join(output_dir, "checkpoints", old_st["dir"])
        shutil.rmtree(stale, ignore_errors=True)


def make_mid_epoch_saver(accelerator, output_dir, epoch, interval=None):
    """为指定 epoch 构造存档回调。engine 每迭代调 saver(completed_steps)；
    内部按频点过滤。返回 None 表示关闭（interval=0）。"""
    interval = MID_EPOCH_SAVE_ITERS if interval is None else interval
    if interval <= 0:
        return None
    anchor = {"last": 0}

    def saver(step):
        try:
            if step - anchor["last"] < interval:
                return
            anchor["last"] = step
            accelerator.save_state(safe_serialization=False)
            new_dir, _ = _newest_checkpoint(output_dir)
            if new_dir is None:
                return
            old_st = _read_state(output_dir)
            _remove_stale_mid_dir(output_dir, old_st, os.path.basename(new_dir))
            _write_state(output_dir, {"kind": "mid", "epoch": epoch, "step": step,
                                      "dir": os.path.basename(new_dir)})
            logger.info(f"[mid-epoch save] epoch {epoch} step {step} -> {os.path.basename(new_dir)}")
        except Exception as e:  # noqa: BLE001 —— 存档失败绝不中断训练
            logger.warning(f"[mid-epoch save] failed (ignored): {e}")

    return saver


def write_epoch_end_state(output_dir, epoch):
    """epoch 边界 save_state() 之后调用：更新标记为 epoch_end 并清理遗留 mid 目录。
    恢复重锚（标记与实际加载目录不一致时）也复用本函数。"""
    try:
        new_dir, _ = _newest_checkpoint(output_dir)
        if new_dir is None:
            return
        old_st = _read_state(output_dir)
        _remove_stale_mid_dir(output_dir, old_st, os.path.basename(new_dir))
        _write_state(output_dir, {"kind": "epoch_end", "epoch": epoch,
                                  "step": 0, "dir": os.path.basename(new_dir)})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[epoch-end state] failed (ignored): {e}")


def resolve_resume(output_dir, loaded_dir_name):
    """恢复推导。loaded_dir_name = 实际加载的 checkpoint 目录名。
    返回 (starting_epoch, skip_batches)。
    - 标记存在且指向加载目录：mid → (epoch, step)；epoch_end → (epoch+1, 0)
    - 标记缺失/不一致：回退历史语义 (目录号+1, 0)，并把标记重锚到加载目录
      （此后新存档接续本标记，目录号不再需要等于 epoch 数）"""
    st = _read_state(output_dir)
    dir_num = int(loaded_dir_name.split("_")[-1])
    if st is not None and st.get("dir") == loaded_dir_name:
        if st["kind"] == "mid":
            return st["epoch"], st["step"]
        return st["epoch"] + 1, 0
    if st is not None:
        logger.warning(f"[resume] state file mismatch (json={st.get('dir')} loaded={loaded_dir_name}) "
                       f"- re-anchoring to loaded dir")
    write_epoch_end_state(output_dir, dir_num)
    return dir_num + 1, 0

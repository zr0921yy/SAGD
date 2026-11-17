import argparse
import datetime
import os
import pprint
import re
import time

import accelerate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import ProjectConfiguration
from torch.utils import data

from util.collate_fn import collate_fn
from util.engine import evaluate_acc, train_one_epoch_acc
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import default_setup, encode_labels, fixed_generator, seed_worker
from util.utils import HighestCheckpoint, load_checkpoint, load_state_dict
from util import train_state


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--accumulate-steps", type=int, default=1, help="Steps to accumulate gradients"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    dynamo_backend = ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser"]
    dynamo_backend += ["cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "ipex", "tvm"]
    parser.add_argument(
        "--dynamo-backend",
        type=str,
        default="no",
        choices=dynamo_backend,
        help="""
        Set to one of the possible dynamo backends to optimize the training with torch dynamo.
        See https://pytorch.org/docs/stable/torch.compiler.html and
        https://huggingface.co/docs/accelerate/main/en/package_reference/utilities#accelerate.utils.DynamoBackend
        """,
    )

    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    cfg = Config(args.config_file, partials=("lr_scheduler", "optimizer", "param_dicts"))

    # modify output directory
    if getattr(cfg, "output_dir", None) is None:
        if hasattr(cfg, "resume_from_checkpoint") and os.path.isdir(str(cfg.resume_from_checkpoint)):
            # RS-Gaze: RESUME_FROM 指向运行目录 = 规范化锚点，两种情形都要让 watchdog 的
            # --run-dir 保持有效（日志始终落在同一运行目录）：
            #   a) 运行目录已有 checkpoint → 续训最新档（损失 ≤1 epoch）
            #   b) 尚无任何 checkpoint（epoch 0 未存档即中断）→ 全新训练但复用同一运行目录
            if "checkpoints" in os.listdir(str(cfg.resume_from_checkpoint)):
                # if given output_dir, find the newest checkpoint under checkpoints directory
                output_dir = os.path.join(str(cfg.resume_from_checkpoint), "checkpoints")
                folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
                folders.sort(
                    key=lambda folder:
                        list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]
                )
                cfg.resume_from_checkpoint = folders[-1]
                # RS-Gaze 半 epoch 存档（util/train_state.py）：train_state.json 是
                # "完整落盘"的提交标记（save_state 完成后才写）——优先选标记指向的
                # 目录，避免恰好选中一个被杀在半写的最新目录；无标记回退最新目录
                # （历史 run dir 无标记，语义不变）。此刻 cfg.output_dir 尚未赋值，
                # json 在运行目录（= checkpoints 的上级）下
                _st = train_state._read_state(os.path.dirname(output_dir))
                if _st is not None:
                    _marked = os.path.join(output_dir, _st["dir"])
                    if os.path.isdir(_marked):
                        cfg.resume_from_checkpoint = _marked
                cfg.output_dir = os.path.dirname(os.path.dirname(str(cfg.resume_from_checkpoint)))
            elif os.path.basename(os.path.normpath(os.path.dirname(str(cfg.resume_from_checkpoint)))) == "checkpoints":
                # RESUME_FROM 直接指向 <运行目录>/checkpoints/checkpoint_N
                cfg.output_dir = os.path.dirname(os.path.dirname(str(cfg.resume_from_checkpoint)))
            else:
                # 情形 b)：目录里没有 checkpoints 子档 → 视为全新训练
                cfg.output_dir = str(cfg.resume_from_checkpoint)
                cfg.resume_from_checkpoint = None
        else:
            # make sure all processes have same output directory
            accelerate.utils.wait_for_everyone()
            cfg.output_dir = os.path.join(
                "checkpoints",
                os.path.basename(cfg.model_path).split(".")[0],
                "train",
                datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
            )

    # ── RS-Gaze: RESUME_FROM 指向运行目录（而非 checkpoint 子目录/权重文件）时，
    #    统一规范化为其中最新 checkpoint 子目录（train_state.json 标记优先）。
    #    与 output_dir 是否预设无关——固定 run dir 的配对实验同样依赖此路径
    #    （watchdog 重启注入 dir 语义；2026-10-20 修复固定 output_dir 时该
    #    规范化被跳过导致 load_state FileNotFoundError 的事故）──
    _rf = getattr(cfg, "resume_from_checkpoint", None)
    if _rf and os.path.isdir(str(_rf)):
        _ckpt_root = os.path.join(str(_rf), "checkpoints")
        if os.path.isdir(_ckpt_root):
            _st = train_state._read_state(str(_rf))
            _marked = os.path.join(_ckpt_root, _st["dir"]) if _st else None
            if _marked and os.path.isdir(_marked):
                cfg.resume_from_checkpoint = _marked
            else:
                _folders = [os.path.join(_ckpt_root, f) for f in os.listdir(_ckpt_root)]
                _folders.sort(key=lambda p: int(re.findall(r"(\d+)$", p)[0]))
                cfg.resume_from_checkpoint = _folders[-1]
        else:
            # 目录里没有任何 checkpoint（如首发即崩的空运行目录）——目录续训
            # 语义不可用，降级为全新训练，避免 load_state 向空目录要文件崩溃
            print(f"[RS-Gaze] WARNING: RESUME_FROM dir has no checkpoints - "
                  f"treating as fresh start: {_rf}")
            cfg.resume_from_checkpoint = None

    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, total_limit=5, automatic_checkpoint_naming=True
    )
    tensorboard_tracker = TensorBoardTracker(run_name="tf_log", logging_dir=cfg.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)
    accelerator = Accelerator(
        log_with=tensorboard_tracker,
        project_config=project_config,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_steps,
        dynamo_backend=args.dynamo_backend,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers("det_train")
    default_setup(args, cfg, accelerator)
    logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)

    # instantiate dataset
    params = dict(num_workers=cfg.num_workers, collate_fn=collate_fn)
    # RS-Gaze: persistent_workers 仅在 num_workers>0 时合法（num_workers=0 = 主线程加载，
    # 用于规避 Windows 多进程 DataLoader 的死锁/静默崩溃类故障）
    params.update(dict(pin_memory=cfg.pin_memory, persistent_workers=cfg.num_workers > 0))
    if args.use_deterministic_algorithms:
        # set using deterministic algorithms
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        params.update({"worker_init_fn": seed_worker, "generator": fixed_generator()})
    # we use group_based sampler, which increases training speed slightly
    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(
        data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
    )
    train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)
    test_loader = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)

    # instantiate model, optimizer and lr_scheduler
    model = Config(cfg.model_path).model
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = cfg.optimizer(cfg.param_dicts(model))
    lr_scheduler = cfg.lr_scheduler(optimizer)

    # load from a pretrained weight and fine-tune on it
    weight_path = getattr(cfg, "resume_from_checkpoint", None)
    if weight_path is not None and os.path.isfile(weight_path):
        checkpoint = load_checkpoint(cfg.resume_from_checkpoint)
        load_state_dict(model, checkpoint)
        logger.info(f"load pretrained from {cfg.resume_from_checkpoint}")

    # register dataset class information into the model, useful for inference
    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # prepare for distributed training
    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )

    # load from a directory, which means resume training
    skip_batches = 0
    if weight_path is not None and os.path.isdir(weight_path):
        accelerator.load_state(cfg.resume_from_checkpoint)
        path = os.path.basename(cfg.resume_from_checkpoint)
        # RS-Gaze 半 epoch 续训推导（util/train_state.py）：train_state.json 指向
        # mid → 本 epoch 第 step 批续跑；epoch_end → 下一 epoch；无标记/不一致 →
        # 回退"目录号+1"历史语义并重锚标记（启用半 epoch 存档后目录号不再等于
        # 已完成 epoch 数，推导一律以标记为准）
        cfg.starting_epoch, skip_batches = train_state.resolve_resume(cfg.output_dir, path)
        accelerator.project_configuration.iteration = int(path.split("_")[-1]) + 1
        logger.info(f"resume training of {cfg.output_dir}, from {path}, "
                    f"epoch {cfg.starting_epoch}, skip {skip_batches} batches")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("model parameters: {}".format(n_params))
        logger.info("optimizer: {}".format(optimizer))
        logger.info("lr_scheduler: {}".format(pprint.pformat(lr_scheduler.state_dict())))

    # save dataset name, useful for inference
    if accelerator.is_main_process:
        label_file = os.path.join(cfg.output_dir, "label_names.txt")
        with open(label_file, "w", encoding="utf-8") as f:
            caid_name = [f"{k} {v['name']}" for k, v in cfg.train_dataset.coco.cats.items()]
            caid_name = "\n".join(caid_name)
            f.write(caid_name)
        logger.info(f"Label names is saved to {label_file}")

    logger.info("Start training")
    # RS-Gaze: 记录训练主进程 PID——watchdog 精确终止训练进程树用
    # （只杀 main.py 及其子进程，不误杀编排器/其他配套 python；2026-10-21 事故修复）
    try:
        with open(os.path.join(cfg.output_dir, "train_pid"), "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass
    start_time = time.perf_counter()
    highest_checkpoint = HighestCheckpoint(accelerator, model)
    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        mid_epoch_saver = train_state.make_mid_epoch_saver(
            accelerator, cfg.output_dir, epoch, train_state.MID_EPOCH_SAVE_ITERS)
        train_one_epoch_acc(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            print_freq=cfg.print_freq,
            max_grad_norm=cfg.max_norm,
            accelerator=accelerator,
            skip_batches=skip_batches,
            mid_epoch_saver=mid_epoch_saver,
        )
        skip_batches = 0  # 半 epoch 跳批只作用于续训的第一个 epoch

        lr_scheduler.step()

        # we save model and labels together
        accelerator.save_state(safe_serialization=False)
        # RS-Gaze: 更新 train_state.json（提交标记指向本 epoch 边界档），并清理
        # 已被推进覆盖的半 epoch 目录
        train_state.write_epoch_end_state(cfg.output_dir, epoch)
        logger.info("Start evaluation")
        coco_evaluator = evaluate_acc(model, test_loader, epoch, accelerator)

        # save best results
        cur_ap, cur_ap50 = coco_evaluator.coco_eval["bbox"].stats[:2]
        highest_checkpoint.update(ap=cur_ap, ap50=cur_ap50)

    total_time = time.perf_counter() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time: {}".format(total_time))
    accelerator.end_training()


if __name__ == "__main__":
    train()

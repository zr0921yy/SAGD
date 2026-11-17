from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 2    # total_batch_size = #GPU x batch_size
num_workers = 2   # workers for pytorch DataLoader（内存混插已拔除、恢复 2×16GB 稳定配置后
                  # 从 0 恢复为 2 以提速；若再遇 DataLoader 挂死/静默退出，改回 0）
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

# SAGD 训练配置；output_dir=None 时 main.py 自动落到
# checkpoints/<模型配置名>/train/<启动时间戳>，多实验可用 RESUME_FROM 续训
output_dir = None
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train（DOTA1.5 切片，16 类；目录结构见 README）
coco_path = "data/DOTA1.5_sliced"  # 相对仓库根，请从仓库根目录启动
coco_ann_train = f"{coco_path}/annotations/instances_train2017.json"
coco_ann_val = f"{coco_path}/annotations/instances_val2017.json"

# RS-Gaze: 人眼注视密度蒸馏三臂开关（环境变量 GAZE_DIR）
#   臂1 基线（框派生监督）: set GAZE_DIR= （清除，或不设置）
#   臂2 伪注视蒸馏:         set GAZE_DIR=data/pseudo_density
#   臂3 真人眼蒸馏:         set GAZE_DIR=data/gaze_density
import os
gaze_dir = os.environ.get("GAZE_DIR", None)
gaze_resolution = 128  # target 中密度图的存储分辨率
print(f"[RS-Gaze] 臂5-v2 选择感知蒸馏: {gaze_dir}")
train_transform = presets.detr  # see transforms/presets to choose a transform
train_dataset = CocoDetection(
    img_folder=coco_path,
    gaze_dir=gaze_dir,
    gaze_resolution=gaze_resolution,
    ann_file=coco_ann_train,
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=coco_path,
    ann_file=coco_ann_val,
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train（DOTA1.5 16 类版）
model_path = "configs/salience_detr/salience_detr_resnet50_dota15_selaware.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
# RS-Gaze: 断点续训开关（环境变量 RESUME_FROM，与 GAZE_DIR 同款约定，同一终端先 set 再启动）
#   续训: set RESUME_FROM=checkpoints/salience_detr_resnet50_dota15_selaware/train/<运行目录>
#   新跑: set RESUME_FROM= （清除，或不设置）
resume_from_checkpoint = os.environ.get("RESUME_FROM", None) or None

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)


# 防呆：main.py 依赖的关键属性缺失时报清晰错误（而非莫名 AttributeError）
for _name in ("model_path", "optimizer", "lr_scheduler", "param_dicts",
              "train_dataset", "test_dataset"):
    assert _name in dir(), f"[train_config] 缺少关键属性: {_name}"

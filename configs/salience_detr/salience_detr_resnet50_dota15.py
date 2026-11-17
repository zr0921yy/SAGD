# Salience DETR — DOTA1.5 16 类配置（RS-Gaze 人眼蒸馏三臂实验用）
# 基于 salience_detr_resnet50_800_1333.py，改动：
#   1. num_classes 91 → 16（DOTA1.5）
#   2. 数据路径指向 DOTA1.5 切片数据（目录结构见 README）
#   3. gaze_dir 通过环境变量 GAZE_DIR 切换三臂：
#        臂1 基线（框派生监督）:   set GAZE_DIR=
#        臂2 伪注视蒸馏:           set GAZE_DIR=data/pseudo_density
#        臂3 真人眼蒸馏:           set GAZE_DIR=data/gaze_density

import os

from torch import nn

from models.backbones.resnet import ResNetBackbone
from models.bricks.misc import FrozenBatchNorm2d
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.salience_transformer import (
    SalienceTransformer,
    SalienceTransformerDecoder,
    SalienceTransformerDecoderLayer,
    SalienceTransformerEncoder,
    SalienceTransformerEncoderLayer,
)
from models.bricks.set_criterion import HybridSetCriterion
from models.detectors.salience_detr import SalienceCriterion, SalienceDETR
from models.matcher.hungarian_matcher import HungarianMatcher
from models.necks.channel_mapper import ChannelMapper
from models.necks.repnet import RepVGGPluXNetwork

# mostly changed parameters
embed_dim = 256
num_classes = 17          # DOTA1.5: 16 类 + 索引 0 空置（category_id 1–16，DETR COCO 同款约定）
num_queries = 900
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

# datasets (DOTA1.5 1024 切片，COCO 格式)
coco_path = "data/DOTA1.5_sliced"  # 相对仓库根，请从仓库根目录启动
coco_ann_train = f"{coco_path}/annotations/instances_train2017.json"
coco_ann_val = f"{coco_path}/annotations/instances_val2017.json"

# RS-Gaze: 人眼蒸馏三臂开关（环境变量 GAZE_DIR；未设置 = 臂1 框派生基线）
gaze_dir = os.environ.get("GAZE_DIR", None) or None
gaze_resolution = 128
if gaze_dir:
    print(f"[RS-Gaze] 蒸馏臂: gaze_dir = {gaze_dir}")
else:
    print("[RS-Gaze] 臂1 基线（框派生监督，无 gaze 蒸馏）")

# instantiate model components
position_embedding = PositionEmbeddingSine(embed_dim // 2, temperature=10000, normalize=True, offset=-0.5)

backbone = ResNetBackbone(
    "resnet50", norm_layer=FrozenBatchNorm2d, return_indices=(1, 2, 3), freeze_indices=(0,)
)

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

transformer = SalienceTransformer(
    encoder=SalienceTransformerEncoder(
        encoder_layer=SalienceTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
    ),
    neck=RepVGGPluXNetwork(
        in_channels_list=neck.num_channels,
        out_channels_list=neck.num_channels,
        norm_layer=nn.BatchNorm2d,
        activation=nn.SiLU,
        groups=4,
    ),
    decoder=SalienceTransformerDecoder(
        decoder_layer=SalienceTransformerDecoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
    level_filter_ratio=(0.4, 0.8, 1.0, 1.0),
    layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
)

matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
weight_dict.update({
    k + f"_{i}": v
    for i in range(transformer_dec_layers - 1)
    for k, v in weight_dict.items()
})
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
weight_dict.update({"loss_salience": 2})

criterion = HybridSetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, alpha=0.25, gamma=2.0)
foreground_criterion = SalienceCriterion(noise_scale=0.0, alpha=0.25, gamma=2.0, gaze_lambda=0.5)
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

# combine above components to instantiate the model
model = SalienceDETR(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    focus_criterion=foreground_criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    min_size=800,
    max_size=1333,
)

from datasets.coco import CocoDetection
from transforms import presets

train_transform = presets.detr
train_dataset = CocoDetection(
    img_folder=coco_path,
    ann_file=coco_ann_train,
    transforms=train_transform,
    train=True,
    gaze_dir=gaze_dir,
    gaze_resolution=gaze_resolution,
)
test_dataset = CocoDetection(
    img_folder=coco_path,
    ann_file=coco_ann_val,
    transforms=None,  # the eval_transform is integrated in the model
)


# 说明：本文件只负责模型与数据集构建；num_epochs/optimizer/lr_scheduler/
# param_dicts 等训练属性由 configs/train_config.py 提供（--config-file 请指向
# train_config.py，main.py 会再以本文件构建模型）

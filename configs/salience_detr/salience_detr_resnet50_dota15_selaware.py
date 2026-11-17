# Salience DETR — DOTA1.5 16 类配置（RS-Gaze 臂5-v2：选择感知蒸馏）
# 基于 salience_detr_resnet50_dota15.py，唯一改动：
#   foreground_criterion 加 sel_ratio_pool=(0.25, 0.35, 0.5)——每个 batch 从池中
#   抽一个部署截断比例，对混合蒸馏目标按"边界度"加权 focal loss（行均值归一，
#   不改损失量级）：0.35 档部署时只有 topk 边界附近的排序决定成败，边界权重把
#   监督火力对准该处。模型结构与臂1/3 完全一致（标准头）。
# 配对实验设计：与 ctrl5（同权重、同 seed、同数据、无加权）对比——
#   两者都从 arm3_s42 best_ap.pth 同源续训（RESUME_FROM 指向文件），
#   Δ(arm5v2 − ctrl5) 为配对读数，统计功效远高于独立种子对比。
# 启动：python main.py --config-file configs/train_config_selaware.py

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

# RS-Gaze: 臂5-v2 与臂3 同数据语义（combined 2被试，由发射环境变量指定）
gaze_dir = os.environ.get("GAZE_DIR", None) or None
gaze_resolution = 128
print(f"[RS-Gaze] 臂5-v2 选择感知蒸馏: gaze_dir = {gaze_dir}")

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
foreground_criterion = SalienceCriterion(noise_scale=0.0, alpha=0.25, gamma=2.0, gaze_lambda=0.5,
                                         sel_ratio_pool=(0.25, 0.35, 0.5))
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

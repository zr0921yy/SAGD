# Salience DETR — DOTA1.5 16 类配置（R50 归因 2×2：目标源 × 边界加权）
# 与 salience_detr_resnet50_dota15_selaware.py 的唯一区别：三个自由度全部由
# 环境变量给定，便于同源 warm start 的受控 2×2：
#   GAZE_DIR          指向真眼动目录（gaze_density）或留空（纯框目标）
#   SEL_POOL          非空（如 "0.25,0.35,0.5"）= 开启边界加权；空 = 关闭
#   SEL_WEIGHT_SPACE  "target"（默认口径，主表实验）或 "pred"（deployment-exact
#                     变体：按预测分数排序，2×2 归因实验用，保证两因子正交）
#   GAZE_LAMBDA       混合权重 λ，默认 0.5
# 模型结构与基线完全一致，推理路径不变。

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
num_classes = 17          # DOTA1.5: 16 类 + 索引 0 空置（category_id 1–16）
num_queries = 900
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

# datasets (DOTA1.5 sliced by tools/slice_dota15.py)
coco_path = "data/DOTA1.5_sliced"
coco_ann_train = f"{coco_path}/annotations/instances_train2017.json"
coco_ann_val = f"{coco_path}/annotations/instances_val2017.json"

# ── 归因实验开关（全部来自环境变量） ──
gaze_dir = os.environ.get("GAZE_DIR", None) or None
gaze_resolution = 128
gaze_lambda = float(os.environ.get("GAZE_LAMBDA", "0.5") or 0.5)
_sel_pool_env = (os.environ.get("SEL_POOL", "0.25,0.35,0.5") or "").strip()
sel_ratio_pool = tuple(float(x) for x in _sel_pool_env.split(",") if x) or None
sel_weight_space = (os.environ.get("SEL_WEIGHT_SPACE", "pred") or "pred").strip()
sel_tau_frac = float(os.environ.get("SEL_TAU_FRAC", "0.10") or 0.10)
print(f"[attrib] gaze_dir={gaze_dir} lambda={gaze_lambda} "
      f"sel_pool={sel_ratio_pool} space={sel_weight_space} tau={sel_tau_frac}")

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
foreground_criterion = SalienceCriterion(
    noise_scale=0.0, alpha=0.25, gamma=2.0, gaze_lambda=gaze_lambda,
    sel_ratio_pool=sel_ratio_pool, sel_tau_frac=sel_tau_frac,
    sel_weight_space=sel_weight_space,
)
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

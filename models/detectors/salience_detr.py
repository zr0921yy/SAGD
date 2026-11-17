from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector


class SalienceCriterion(nn.Module):
    def __init__(
        self,
        limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
        noise_scale: float = 0.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        gaze_lambda: float = 0.5,
        sel_ratio_pool: Tuple = None,
        sel_tau_frac: float = 0.10,
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma
        self.gaze_lambda = gaze_lambda
        # ── RS-Gaze 臂5-v2 选择感知蒸馏：非空时每个 batch 从池中抽一个部署截断
        #    比例，对混合目标按"边界度"加权 focal loss——0.35 档部署时只有 topk
        #    边界附近的排序决定成败，边界权重把监督火力对准该处（行均值归一，
        #    不改变损失量级）。None = 关闭（臂1/3 原行为，逐位一致）。
        self.sel_ratio_pool = sel_ratio_pool
        self.sel_tau_frac = sel_tau_frac

    def forward(self, foreground_mask, targets, feature_strides, image_sizes,
                gaze_lambda: float = 0.5):
        gt_boxes_list = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(boxes * scale_factor)

        mask_targets = []
        for level_idx, (mask, feature_stride) in enumerate(zip(foreground_mask, feature_strides)):
            feature_shape = mask.shape[-2:]
            coord_x, coord_y = self.get_pixel_coordinate(feature_shape, feature_stride, device=mask.device)
            masks_per_level = []
            for gt_boxes in gt_boxes_list:
                mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, level_idx)
                masks_per_level.append(mask)

            masks_per_level = torch.stack(masks_per_level)
            mask_targets.append(masks_per_level)
        # ── RS-Gaze 人眼蒸馏（IPM）：框派生目标与人眼密度目标加权混合 ──
        # targets[i]["gaze_density"]: (1, H, W) 或 (1, 1, H, W)（数据端 Mask 包装）
        # 逐图门控：眼动只覆盖约 8% 训练 tile，若按 batch 级开关混合，
        # 同 batch 中 92% 无眼动图会被一并压成 (1-λ)·mask_gt（默认 λ=0.5，
        # 等于把它们的前景监督砍半）。因此按图判定，无眼动图保持纯框监督。
        has_gaze = any("gaze_density" in t for t in targets)
        level_sizes = [m.shape[1] for m in mask_targets]   # 供边界加权按层切分
        mask_gt = torch.cat(mask_targets, dim=1)
        if has_gaze and gaze_lambda > 0:
            gate = torch.tensor(
                [1.0 if t.get("gaze_density") is not None else 0.0 for t in targets],
                device=mask_gt.device, dtype=mask_gt.dtype).view(-1, 1)
            gaze_targets = []
            for level_idx, (mask, feature_stride) in enumerate(
                    zip(foreground_mask, feature_strides)):
                feature_shape = mask.shape[-2:]
                gaze_level = []
                for t in targets:
                    g = t.get("gaze_density")
                    if g is not None and g.dim() == 4:   # (1,1,H,W) → (1,H,W)
                        g = g[0]
                    if g is None:
                        g = torch.zeros(
                            (1, *feature_shape), device=mask.device,
                            dtype=mask.dtype)
                    else:
                        g = F.interpolate(
                            g[None].to(mask.device, mask.dtype),
                            size=feature_shape, mode="bilinear",
                            align_corners=True)[0]
                        if g.max() > 0:      # 逐图 max 归一化（密度分布 → 打分目标）
                            g = g / g.max()
                    gaze_level.append(g.flatten())                     # (h*w,) 与 mask_targets 展平格式对齐
                gaze_targets.append(torch.stack(gaze_level))
            gaze_targets = torch.cat(gaze_targets, dim=1)
            mask_targets = mask_gt + gate * gaze_lambda * (gaze_targets - mask_gt)
        else:
            mask_targets = mask_gt

        foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
        foreground_mask = foreground_mask.squeeze(1)
        # num_pos 仍由 GT 框目标决定（保持正样本语义稳定，不随蒸馏目标软化）
        num_pos = torch.sum(mask_gt > 0.5 * self.noise_scale).clamp_(min=1)
        sel_weights = None
        if self.sel_ratio_pool:
            r = self.sel_ratio_pool[int(torch.randint(len(self.sel_ratio_pool), (1,)).item())]
            sel_weights = []
            start = 0
            for size in level_sizes:
                sel_weights.append(
                    self._boundary_weights(mask_targets[:, start:start + size], r,
                                           tau_frac=self.sel_tau_frac))
                start += size
            sel_weights = torch.cat(sel_weights, dim=1)
        salience_loss = (
            sigmoid_focal_loss(
                foreground_mask,
                mask_targets,
                num_pos,
                alpha=self.alpha,
                gamma=self.gamma,
                weights=sel_weights,
            ) * foreground_mask.shape[1]
        )
        out = {"loss_salience": salience_loss}
        if has_gaze and gaze_lambda > 0:
            out["gaze_lambda"] = gaze_lambda      # 记录，供日志/消融
        return out

    @staticmethod
    def _boundary_weights(tgt, ratio: float, tau_frac: float = 0.10):
        """选择感知边界加权（臂5-v2）：按混合目标降序排名，在部署截断点
        k = T·ratio 附近放指数核权重峰（半宽 tau = 0.10·k），行均值归一 →
        只改监督重点不改损失量级。目标为 0 的 token（含 pad）排名靠后，权重趋零。"""
        B, T = tgt.shape
        k = min(max(float(T) * min(max(ratio, 0.01), 1.0), 1.0), float(T))
        order = tgt.argsort(dim=1, descending=True)
        rank = torch.empty_like(order, dtype=tgt.dtype)
        bidx = torch.arange(B, device=tgt.device).unsqueeze(1)
        rank[bidx, order] = torch.arange(T, device=tgt.device, dtype=tgt.dtype).unsqueeze(0)
        tau = max(1.0, tau_frac * k)
        w = torch.exp(-((rank - k).abs()) / tau)
        return w / w.mean(dim=1, keepdim=True).clamp_min(1e-6)

    def get_pixel_coordinate(self, feature_shape, stride, device):
        height, width = feature_shape
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
            indexing="ij",
        )
        coord_y = coord_y.reshape(-1)
        coord_x = coord_x.reshape(-1)
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
        # gt_label: (m,) gt_boxes: (m, 4)
        # coord_x: (h*w, )
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )  # [h*w, m, 4]

        # the foreground queries must satisfy two requirements:
        # 1. the quereis located in bounding boxes
        # 2. the distance from queries to the box center match the feature map stride
        min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m]
        max_border_distances = torch.max(border_distances, dim=-1)[0]
        mask_in_gt_boxes = min_border_distances > 0
        min_limit, max_limit = self.limit_range[level_idx]
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
        mask_pos = mask_in_gt_boxes & mask_in_level

        # scale-independent salience confidence
        row_factor = left_border_distance + right_border_distance
        col_factor = top_border_distance + bottom_border_distance
        delta_x = (left_border_distance - right_border_distance) / row_factor
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

        confidence_per_box = 1 - confidence
        confidence_per_box[~mask_in_gt_boxes] = 0

        # process positive coordinates
        if confidence_per_box.numel() != 0:
            mask = confidence_per_box.max(-1)[0]
        else:
            mask = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

        # process negative coordinates
        mask_pos = mask_pos.long().sum(dim=-1) >= 1
        mask[~mask_pos] = 0

        # add noise to add randomness
        mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
        return mask


# SalienceDETR has the architecture similar to FocusDETR
class SalienceDETR(DNDETRDetector):
    def __init__(
        # model structure
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        focus_criterion: nn.Module,
        gaze_lambda: float = 0.5,
        # model parameters
        num_classes: int = 91,
        num_queries: int = 900,
        denoising_nums: int = 100,
        # model variants
        aux_loss: bool = True,
        min_size: int = None,
        max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.denoising_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=embed_dim,
            denoising_nums=denoising_nums,
            label_noise_prob=0.5,
            box_noise_scale=1.0,
        )
        self.focus_criterion = focus_criterion
        self.gaze_lambda = gaze_lambda

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # extract features
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.neck(multi_level_feats)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            noised_results = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_query = noised_results[0]
            noised_box_query = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_query = None
            noised_box_query = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None

        # feed into transformer
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )
        # hack implementation for distributed training
        outputs_class[0] += self.denoising_generator.label_encoder.weight[0, 0] * 0.0

        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

            # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # compute focus loss
            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]
            focus_loss = self.focus_criterion(
                foreground_mask, targets, feature_stride, images.image_sizes,
                gaze_lambda=getattr(self, "gaze_lambda", 0.5))
            loss_dict.update(focus_loss)

            # ── RS-Gaze 臂5：自适应先验头的路由均衡正则。守卫式：其余臂的
            #    transformer 没有 take_routing_loss 属性，此处零影响；
            #    是否计入总损失由 config 的 weight_dict 是否含 loss_routing 决定 ──
            take_routing_loss = getattr(self.transformer, "take_routing_loss", None)
            if take_routing_loss is not None:
                routing_loss = take_routing_loss()
                if routing_loss is not None:
                    loss_dict["loss_routing"] = routing_loss

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k]) for k in loss_dict.keys() if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections

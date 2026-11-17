import csv
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision

from transforms import v2 as T
from transforms.convert_coco_polys_to_mask import ConvertCocoPolysToMask
from util import datapoints
from util.misc import deepcopy


def load_gaze_split_stems(split_file, split: str = "train"):
    """读取 gaze_split.csv，返回该 split 的眼动 stem 集合（训练白名单）。"""
    stems = set()
    with open(split_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                stems.add(row["stem"])
    return stems


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        train=False,
        gaze_dir: str = None,
        gaze_resolution: int = 128,
        gaze_split_file: str = None,
        gaze_split: str = "train",
        gaze_task: str = "search",
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.prepare = ConvertCocoPolysToMask()
        self._transforms = transforms
        self._transforms = self.update_dataset(self._transforms)
        self.train = train
        # RS-Gaze: 人眼注视密度图目录（npz，1024×1024 float16，stem 命名）
        # 硬约束：眼动只作 train 侧辅助监督，val/test 一律断开（即使调用方误传
        # gaze_dir）；gaze_split_file 给定时再限制在 train 白名单内，双重守卫。
        self.gaze_dir = Path(gaze_dir) if gaze_dir and train else None
        self.gaze_allowed = None
        if self.gaze_dir is not None and gaze_split_file is not None:
            self.gaze_allowed = load_gaze_split_stems(gaze_split_file, gaze_split)
        # 同一 stem 可能同时有目标搜索密度与自由观看密度（184 张 tile 两个任务
        # 都看过），按 stem 建索引会互相覆盖，所以按任务子目录分别索引
        self.gaze_task = gaze_task
        self.gaze_resolution = gaze_resolution

        if train:
            self._coco_remove_images_without_annotations()

    def update_dataset(self, transform):
        if isinstance(transform, (T.Compose, A.Compose)):
            processed_transforms = []
            for trans in transform.transforms:
                trans = self.update_dataset(trans)
                processed_transforms.append(trans)
            return type(transform)(processed_transforms)
        if hasattr(transform, "update_dataset"):
            transform.update_dataset(self)
        return transform

    def load_image(self, image_name):
        # after comparing the speed of PIL, torchvision and cv2,
        # cv2 is chosen as the default backend to load images,
        # uncomment the following code to switch among them.

        # image = Image.open(os.path.join(self.root, path)).convert('RGB')
        # image = torchvision.io.read_image(os.path.join(self.root, path))

        # To avoid deadlock between DataLoader and OpenCV
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # image = cv2.imread(os.path.join(self.root, image_name))
        image = cv2.imdecode(np.fromfile(os.path.join(self.root, image_name), dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return image

    def get_image_id(self, item: int):
        if hasattr(self, "indices"):
            item = self.indices[item]
        image_id = self.ids[item]
        return image_id

    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        self._current_image_name = image_name
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def _gaze_index(self):
        """RS-Gaze: 惰性构建 stem→npz 路径索引（stem 全局唯一，不依赖子目录结构）。"""
        if getattr(self, "_gaze_index_cache", None) is None:
            idx = {}
            if self.gaze_dir is not None:
                base = Path(self.gaze_dir)
                task_dir = base / self.gaze_task
                if task_dir.is_dir():
                    base = task_dir
                for f in sorted(base.rglob("*.npz")):
                    idx.setdefault(f.stem, f)
            self._gaze_index_cache = idx
        return self._gaze_index_cache

    def load_gaze_density(self, image_name):
        """RS-Gaze: 按 stem 加载人眼注视密度图，返回 (1, 1, res, res) float 张量；
        非眼动覆盖 tile / 非 train split 返回 None。"""
        if self.gaze_dir is None:
            return None
        stem = Path(image_name).stem
        if self.gaze_allowed is not None and stem not in self.gaze_allowed:
            return None
        p = self._gaze_index().get(stem)
        if p is None:
            return None
        d = np.load(p)["density"].astype(np.float32)
        if d.max() > 0:
            d = d / d.max()
        return torch.from_numpy(d)[None][None]   # (1,1,H,W)

    def data_augmentation(self, image, target):
        # preprocess
        image = datapoints.Image(image)
        bounding_boxes = datapoints.BoundingBox(
            target["boxes"],
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=image.shape[-2:],
        )
        labels = target["labels"]
        # RS-Gaze: 密度图包装为 datapoints.Mask，v2 几何变换自动同步
        # （flip/crop/scalejitter），训练后下采样到 gaze_resolution 存入 target
        gaze_mask = None
        if self.gaze_dir is not None:
            g = self.load_gaze_density(self._current_image_name)
            if g is not None:
                gaze_mask = datapoints.Mask(
                    torch.nn.functional.interpolate(
                        g, size=tuple(image.shape[-2:]), mode="bilinear",
                        align_corners=True),
                )
        if self._transforms is not None:
            if gaze_mask is not None:
                image, bounding_boxes, gaze_mask, labels = self._transforms(
                    image, bounding_boxes, gaze_mask, labels)
            else:
                image, bounding_boxes, labels = self._transforms(image, bounding_boxes, labels)

        if gaze_mask is not None:
            gaze_mask = torch.nn.functional.interpolate(
                gaze_mask, size=(self.gaze_resolution,) * 2,
                mode="bilinear", align_corners=True)
            target["gaze_density"] = gaze_mask

        return image.data, bounding_boxes.data, labels

    def __getitem__(self, item):
        image, target = self.load_image_and_target(item)
        image, target["boxes"], target["labels"] = self.data_augmentation(image, target)

        return deepcopy(image), deepcopy(target)

    def __len__(self):
        return len(self.indices) if hasattr(self, "indices") else len(self.ids)

    def _coco_remove_images_without_annotations(self, cat_list=None):
        def _has_only_empty_bbox(anno):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

        def _count_visible_keypoints(anno):
            return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

        min_keypoints_per_image = 10

        def _has_valid_annotation(anno):
            # if it's empty, there is no annotation
            if len(anno) == 0:
                return False
            # if all boxes have close to zero area, there is no annotation
            if _has_only_empty_bbox(anno):
                return False
            # keypoints task have a slight different critera for considering
            # if an annotation is valid
            if "keypoints" not in anno[0]:
                return True
            # for keypoint detection tasks, only consider valid images those
            # containing at least min_keypoints_per_image
            if _count_visible_keypoints(anno) >= min_keypoints_per_image:
                return True
            return False

        ids = []
        for ds_idx, img_id in enumerate(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if cat_list:
                anno = [obj for obj in anno if obj["category_id"] in cat_list]
            if _has_valid_annotation(anno):
                ids.append(ds_idx)

        self.indices = ids


class Object365Detection(CocoDetection):
    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        # NOTE: Only for object 365
        image_name = os.path.join(*image_name.split(os.sep)[-2:])
        if self.train:
            image_name = os.path.join("images/train", image_name)
        else:
            image_name = os.path.join("images/val", image_name)
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def __getitem__(self, item):
        try:
            image, target = self.load_image_and_target(item)
        except:
            item += 1
            image, target = self.load_image_and_target(item)
        image, target["boxes"], target["labels"] = self.data_augmentation(image, target)

        return deepcopy(image), deepcopy(target)

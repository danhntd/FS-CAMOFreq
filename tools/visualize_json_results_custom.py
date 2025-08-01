#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from detectron2.data.datasets import register_coco_instances



# register data
register_coco_instances("camo_test_novel1", {}, "/mmlabworkspace/WorkSpaces/danhnt/danhnt/camo/Annotations/camosplit6_noset/camo5_test_split1.json", "/mmlabworkspace/WorkSpaces/danhnt/danhnt/camo/images/")

camopp_test_metadata = MetadataCatalog.get("camo_test_novel1")
camopp_test_dataset_dicts = DatasetCatalog.get("camo_test_novel1")

print(camopp_train_dataset_dicts[0]['file_name'])
print(camopp_train_metadata)


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
    print(bbox.shape[0])
    if bbox.shape[0] > 0: 
      bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", default="/mmlabworkspace/WorkSpaces/danhnt/danhnt/iFS-RCNN/checkpoints/camo_ifs_default/camo_ifs_default_noset_origin/camo_model_novel1_5shot_mask_rcnn_R_101_FPN_ifs/inference/coco_instances_results.json", help="JSON file produced by the model")
    parser.add_argument("--output", default="/mmlabworkspace/WorkSpaces/danhnt/danhnt/iFS-RCNN/checkpoints/camo_ifs_default/camo_ifs_default_noset_origin/camo_model_novel1_5shot_mask_rcnn_R_101_FPN_ifs/visualization/", help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="camo_test_novel1")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    print(pred_by_image)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
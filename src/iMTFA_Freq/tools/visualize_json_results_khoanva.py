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
from detectron2.utils.fs_visualizer import Visualizer, ColorMode


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
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
    parser.add_argument("--input",  default="", help="JSON file produced by the model")
    parser.add_argument("--inputs", default=["/mmlabworkspace/WorkSpaces/danhnt/danhnt/iMTFA/checkpoints/camo/camo_model_test_mask_rcnn_R_101_FPN_normalized_novel1_5shot/inference/coco_instances_results.json"], nargs="+", help="JSONs file produced by the model")
    parser.add_argument("--output", default="/mmlabworkspace/WorkSpaces/danhnt/danhnt/iMTFA/visualization/", help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="camo_test_novel1")
    parser.add_argument("--conf-threshold", default=0.06, type=float, help="confidence threshold")
    parser.add_argument("--phase", default='novel', type=str, help="")
    args = parser.parse_args()

    logger = setup_logger()
    
    if len(args.inputs) > 0:
        print(args.inputs)
        inputs = tuple(args.inputs)
        print(inputs)
        print(type(inputs))
        set_predictions = []
        for input in inputs:       
            print(input) 
            with PathManager.open(input, "r") as f:
                predictions = json.load(f)

            pred_by_image = defaultdict(list)
            for p in predictions:
                pred_by_image[p["image_id"]].append(p)

            set_predictions.append(pred_by_image)
        


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
        novel_id = metadata.novel_dataset_id_to_contiguous_id.keys() 
        print('novel_id:', novel_id)
        print(metadata.thing_classes)
        print(metadata.novel_classes)
        # assert 0
        for dic in tqdm.tqdm(dicts):
            try:
                img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
                basename = os.path.basename(dic["file_name"])
                annos = dic.get("annotations", None)
                print('basename', basename)
                print(annos)
                if args.phase=='novel':
                    if annos:
                        dic_classes = [x["category_id"] in novel_id for x in annos]
                        if all(dic_classes) == False: 
                            print(novel_id)
                            print([x["category_id"] for x in annos])
                            assert 0
                            continue
                    else: 
                        assert 0
                        continue

                vis = Visualizer(img, metadata, instance_mode=ColorMode.SEGMENTATION)
                vis_gt = vis.draw_dataset_dict(dic).get_image()

                concat = [vis_gt]

                for pred_by_image in set_predictions:

                    predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
                    vis = Visualizer(img, metadata, instance_mode=ColorMode.SEGMENTATION)
                    vis_pred = vis.draw_instance_predictions(predictions).get_image()
                    concat.append(vis_pred)

                concat = np.concatenate(concat, axis=0)
                cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])

            except:
                pass


    else:  
        with PathManager.open(args.input, "r") as f:
            predictions = json.load(f)

        pred_by_image = defaultdict(list)
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
            vis = Visualizer(img, metadata, instance_mode=ColorMode.SEGMENTATION)
            # vis = Visualizer(img, metadata)
            vis_pred = vis.draw_instance_predictions(predictions).get_image()

            vis = Visualizer(img, metadata, instance_mode=ColorMode.SEGMENTATION)
            vis_gt = vis.draw_dataset_dict(dic).get_image()

            concat = np.concatenate((vis_pred, vis_gt), axis=1)
            cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])

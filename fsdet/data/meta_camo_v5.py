import numpy as np
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
import copy
import contextlib
import io
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


__all__ = ["register_meta_camo_v5"]


def load_camo_json(json_file, image_root, metadata, dataset_name, sid):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    print('go to here')
    max_shot = {
        1 : 5,
        2 : 10,
        3 : 10,
    }[sid]

    is_shots = "shot" in dataset_name
    if is_shots:
        fileids = {}
        #split_dir = os.path.join("datasets", "camo/Annotations/camosplit3/globalsplit", 'split{}'.format(sid))
        split_dir = os.path.join("datasets", "camo/Annotations/camosplit4/subsplit", 'split{}'.format(sid))
        split_dir = os.path.join("datasets", "camo/Annotations/camosplit6_noset/subsplit", 'split{}'.format(sid))

        # get shot for few-shot learning
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1].split("shot")[0]

        # get instance
        for idx, cls in enumerate(metadata["thing_classes"]):
            json_file = os.path.join(
                # split_dir, "camo{}_{}_{}shot_split{}.json".format(max_shot, cls, max_shot, sid) # for global
                split_dir, "camo{}_{}_{}shot_split{}.json".format(max_shot, cls, shot, sid)
            )

            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    else:
        # print(json_file)
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))

    id_map = metadata["thing_dataset_id_to_contiguous_id"]

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]
    num_instances_without_valid_segmentation = 0
    if is_shots:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, f"{shot}shot_aug", img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    segm = anno.get("segmentation", None)
                    if segm:  # either list[list[float]] or dict(RLE)
                        if not isinstance(segm, dict):
                            # filter out invalid polygons (< 3 points)
                            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                            if len(segm) == 0:
                                num_instances_without_valid_segmentation += 1
                                continue  # ignore this instance
                        obj["segmentation"] = segm

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                # dicts = np.random.choice(dicts, int(shot), replace=False)
                dicts = dicts[:int(shot)]

            for i in range(len(dicts)):
                file_name = dicts[i]["file_name"]
                dataset_dicts.append(dicts[i])
                dicts[i]["file_name"] = file_name.replace(".jpg", "_v5.jpg")
                dataset_dicts.append(dicts[i])
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                segm = anno.get("segmentation", None)
                if segm:  # either list[list[float]] or dict(RLE)
                    if not isinstance(segm, dict):
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        print(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def register_meta_camo_v5(name, metadata, imgdir, annofile, sid):
    metadata = metadata[sid]

    DatasetCatalog.register(
        name,
        lambda: load_camo_json(annofile, imgdir, metadata, name, sid),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="camo",
        dirname="datasets/camo/images",
        **metadata,
    )


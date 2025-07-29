"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Here we only register the few-shot datasets and complete COCO, PascalVOC and
LVIS have been handled by the builtin datasets in detectron2.
"""

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import (
    get_lvis_instances_meta,
    register_lvis_instances,
)
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets.register_coco import register_coco_instances

from .builtin_meta import _get_builtin_metadata
from .meta_coco import register_meta_coco
from .meta_lvis import register_meta_lvis
from .meta_pascal_voc import register_meta_pascal_voc
from .meta_camo import register_meta_camo

from .meta_camo_v4 import register_meta_camo_v4
from .meta_camo_v5 import register_meta_camo_v5
from .meta_camo_v4_only import register_meta_camo_v4_only
from .meta_camo_v5_only import register_meta_camo_v5_only

from .meta_camo_blendeddiff_v0 import register_meta_camo_blendeddiff_v0
from .meta_camo_diffinpainting_v0 import register_meta_camo_diffinpainting_v0
from .meta_camo_gligen_v0 import register_meta_camo_gligen_v0

from .meta_camo_blendeddiff_v1_histogramfiltering import register_meta_camo_blendeddiff_v1_histogramfiltering
from .meta_camo_diffinpainting_v1_histogramfiltering import register_meta_camo_diffinpainting_v1_histogramfiltering
from .meta_camo_gligen_v1_histogramfiltering import register_meta_camo_gligen_v1_histogramfiltering

from .meta_camo_blendeddiff_v2_histogrammatching import register_meta_camo_blendeddiff_v2_histogrammatching
from .meta_camo_diffinpainting_v2_histogrammatching import register_meta_camo_diffinpainting_v2_histogrammatching
from .meta_camo_gligen_v2_histogrammatching import register_meta_camo_gligen_v2_histogrammatching

from .meta_camo_blendeddiff_v3_histogrammatching_1real_4syn import register_meta_camo_blendeddiff_v3_histogrammatching_1real_4syn
from .meta_camo_diffinpainting_v3_histogrammatching_1real_4syn import register_meta_camo_diffinpainting_v3_histogrammatching_1real_4syn
from .meta_camo_gligen_v3_histogrammatching_1real_4syn import register_meta_camo_gligen_v3_histogrammatching_1real_4syn

from .meta_camo_blendeddiff_v4_histogrammatching_1real_4syn_noleak import register_meta_camo_blendeddiff_v4_histogrammatching_1real_4syn_noleak
from .meta_camo_diffinpainting_v4_histogrammatching_1real_4syn_noleak import register_meta_camo_diffinpainting_v4_histogrammatching_1real_4syn_noleak
from .meta_camo_gligen_v4_histogrammatching_1real_4syn_noleak import register_meta_camo_gligen_v4_histogrammatching_1real_4syn_noleak

from .meta_camo_datazoo_v5_histogrammatching_1real_4syn_noleak import register_meta_camo_datazoo_v5_histogrammatching_1real_4syn_noleak
from .meta_camo_datazoo_v5_1real_4syn_noleak import register_meta_camo_datazoo_v5_1real_4syn_noleak

from .meta_camo_blendeddiff_v5_background_1real_4syn_noleak import register_meta_camo_blendeddiff_v5_background_1real_4syn_noleak
from .meta_camo_diffinpainting_v5_background_1real_4syn_noleak import register_meta_camo_diffinpainting_v5_background_1real_4syn_noleak
from .meta_camo_gligen_v5_background_1real_4syn_noleak import register_meta_camo_gligen_v5_background_1real_4syn_noleak

from .meta_camo_blendeddiff_v6_background_1real_1syn_noleak import register_meta_camo_blendeddiff_v6_background_1real_1syn_noleak
from .meta_camo_diffinpainting_v6_background_1real_1syn_noleak import register_meta_camo_diffinpainting_v6_background_1real_1syn_noleak
from .meta_camo_gligen_v6_background_1real_1syn_noleak import register_meta_camo_gligen_v6_background_1real_1syn_noleak

from .meta_camo_blendeddiff_v7_background_1real_1syn_noleak_noinstances import register_meta_camo_blendeddiff_v7_background_1real_1syn_noleak_noinstances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": (
        "coco/train2014",
        "coco/annotations/instances_train2014.json",
    ),
    "coco_2014_val": (
        "coco/val2014",
        "coco/annotations/instances_val2014.json",
    ),
    "coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/instances_minival2014.json",
    ),
    "coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/instances_minival2014_100.json",
    ),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": (
        "coco/train2017",
        "coco/annotations/instances_train2017.json",
    ),
    "coco_2017_val": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
    ),
    "coco_2017_test": (
        "coco/test2017",
        "coco/annotations/image_info_test2017.json",
    ),
    "coco_2017_test-dev": (
        "coco/test2017",
        "coco/annotations/image_info_test-dev2017.json",
    ),
    "coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/instances_val2017_100.json",
    ),
}


def register_all_coco(root="datasets"):
    # for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
    #     for key, (image_root, json_file) in splits_per_dataset.items():
    #         # Assume pre-defined datasets live in `./datasets`.
    #         register_coco_instances(
    #             key,
    #             _get_builtin_metadata(dataset_name),
    #             os.path.join(root, json_file)
    #             if "://" not in json_file
    #             else json_file,
    #             os.path.join(root, image_root),
    #         )

    # register meta datasets
    METASPLITS = [
        (
            "coco_trainval_all",
            "coco/trainval2014",
            "cocosplit/datasplit/trainvalno5k.json",
        ),
        (
            "coco_trainval_base",
            "coco/trainval2014",
            "cocosplit/datasplit/trainvalno5k.json",
        ),
        ("coco_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed)
                # METASPLITS.append((name, "coco/trainval2014", "")) # ori
                METASPLITS.append((name, "coco/trainval2014",  "cocosplit/datasplit/5k.json"))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "lvis_v0.5": {
        # "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_train_freq": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_freq.json",
        ),
        "lvis_v0.5_train_common": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_common.json",
        ),
        "lvis_v0.5_train_rare": (
            "coco/train2017",
            "lvis/lvis_v0.5_train_rare.json",
        ),
        #"lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        #"lvis_v0.5_val_rand_100": (
        #    "coco/val2017",
        #    "lvis/lvis_v0.5_val_rand_100.json",
        #),
        #"lvis_v0.5_test": (
        #    "coco/test2017",
        #    "lvis/lvis_v0.5_image_info_test.json",
        #),
    },
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,
                os.path.join(root, image_root),
            )

    # register meta datasets
    METASPLITS = [
        (
            "lvis_v0.5_train_shots",
            "coco/train2017",
            "lvissplit/lvis_shots.json",
        ),
        (
            "lvis_v0.5_train_rare_novel",
            "coco/train2017",
            "lvis/lvis_v0.5_train_rare.json",
        ),
        #("lvis_v0.5_val_novel", "coco/val2017", "lvis/lvis_v0.5_val.json"),
    ]

    for name, image_root, json_file in METASPLITS:
        dataset_name = "lvis_v0.5_fewshot" if "novel" in name else "lvis_v0.5"
        register_meta_lvis(
            name,
            _get_builtin_metadata(dataset_name),
            os.path.join(root, json_file)
            if "://" not in json_file
            else json_file,
            os.path.join(root, image_root),
        )

# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    # SPLITS = [
    #     ("voc_2007_trainval", "VOC2007", "trainval"),
    #     ("voc_2007_train", "VOC2007", "train"),
    #     ("voc_2007_val", "VOC2007", "val"),
    #     ("voc_2007_test", "VOC2007", "test"),
    #     ("voc_2012_trainval", "VOC2012", "trainval"),
    #     ("voc_2012_train", "VOC2012", "train"),
    #     ("voc_2012_val", "VOC2012", "val"),
    # ]
    # for name, dirname, split in SPLITS:
    #     year = 2007 if "2007" in name else 2012
    #     register_pascal_voc(name, os.path.join(root, dirname), split, year)
    #     MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    # register meta datasets
    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(100):
                        seed = "" if seed == 0 else "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(
            name,
            _get_builtin_metadata("pascal_voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_camo(root="datasets"):
    # register meta datasets
    METASPLITS = [
        # ("camo_train_base", "camo/images", "datasplit/re_noncamo_train.json"),
        ("camo_test_novel1", "camo/images", "camo/Annotations/camosplit6_noset/camo5_test_split1.json", 1),
        # ("camo_test_novel2", "camo/images", "camo/Annotations/camosplit6_noset/camo10_test_split2.json", 2),
        # ("camo_test_novel3", "camo/images", "camo/Annotations/camosplit6_noset/camo10_test_split3.json", 3),

        # ("camo_test_novel2", "camo/images", "datasplit/camo_test_novel2.json", 2),
        # ("camo_test_novel3", "camo/images", "datasplit/camo_test_novel3.json", 3),

        # ("coco_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        # ("coco_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        # ("coco_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]

    format_template = 'datasplit/camo_train_novel{}.json' # will be ignored in the register_meta_camo() function

    # register small meta datasets for fine-tuning stage
    for prefix in ["novel"]:
        for sid in [1, 2, 3]:
            for shot in [1, 2, 3, 5, 10]:
                name = "camo_train_{}{}_{}shot".format(prefix, sid, shot)
                annofile = format_template.format(sid)
                # METASPLITS.append((name, "camo/images", "datasplit/camo_train_novel3.json", sid))
                METASPLITS.append((name, "camo/images", annofile, sid))


                for aug_ver in ["v4", "v4_only", "v5", "v5_only"]:
                    name = "camo_train_{}_{}{}_{}shot".format(aug_ver, prefix, sid, shot)
                    annofile = format_template.format(sid)
                    METASPLITS.append((name, "camo/images", annofile, sid))

    for name, imgdir, annofile, sid in METASPLITS:
        if "camo_train_v4_only_" in name:
            register_meta_camo_v4_only(
                name,
                _get_builtin_metadata("camo_fewshot"),
                os.path.join(root, imgdir),
                os.path.join(root, annofile),
                sid,
            )
        elif "camo_train_v4_" in name:
            register_meta_camo_v4(
                name,
                _get_builtin_metadata("camo_fewshot"),
                os.path.join(root, imgdir),
                os.path.join(root, annofile),
                sid,
            )
        elif "camo_train_v5_only_" in name:
            register_meta_camo_v5_only(
                name,
                _get_builtin_metadata("camo_fewshot"),
                os.path.join(root, imgdir),
                os.path.join(root, annofile),
                sid,
            )
        elif "camo_train_v5_" in name:
            register_meta_camo_v5(
                name,
                _get_builtin_metadata("camo_fewshot"),
                os.path.join(root, imgdir),
                os.path.join(root, annofile),
                sid,
            )
        else:
            register_meta_camo(
                name,
                _get_builtin_metadata("camo_fewshot"),
                os.path.join(root, imgdir),
                os.path.join(root, annofile),
                sid,
            )

    format_template = 'camosplit/camo_train_novel{}.json' # will be ignored in the register_meta_camo() function

    # register small meta datasets for fine-tuning stage of "blendeddiff_v0", "diffinpainting_v0", "gligen_v0"
    for syn_ver in ["blendeddiff_v0", "diffinpainting_v0", "gligen_v0",
    "blendeddiff_v1_histogramfiltering", "diffinpainting_v1_histogramfiltering", "gligen_v1_histogramfiltering",
    "blendeddiff_v2_histogrammatching", "diffinpainting_v2_histogrammatching", "gligen_v2_histogrammatching",
    "blendeddiff_v3_histogrammatching_1real_4syn", "diffinpainting_v3_histogrammatching_1real_4syn", "gligen_v3_histogrammatching_1real_4syn",
    "blendeddiff_v4_histogrammatching_1real_4syn_noleak", "diffinpainting_v4_histogrammatching_1real_4syn_noleak", "gligen_v4_histogrammatching_1real_4syn_noleak",
    "datazoo_v5_histogrammatching_1real_4syn_noleak",
    "datazoo_v5_1real_4syn_noleak",
    "blendeddiff_v5_background_1real_4syn_noleak", "diffinpainting_v5_background_1real_4syn_noleak", "gligen_v5_background_1real_4syn_noleak",
    "blendeddiff_v6_background_1real_1syn_noleak", "diffinpainting_v6_background_1real_1syn_noleak", "gligen_v6_background_1real_1syn_noleak",
    "blendeddiff_v7_background_1real_1syn_noleak_noinstances"
    ]:
        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [1, 2, 3, 5]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images", annofile, sid))

        if syn_ver == "blendeddiff_v0":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v0(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "diffinpainting_v0":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_diffinpainting_v0(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "gligen_v0":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_gligen_v0(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "blendeddiff_v1_histogramfiltering":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v1_histogramfiltering(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "diffinpainting_v1_histogramfiltering":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_diffinpainting_v1_histogramfiltering(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "gligen_v1_histogramfiltering":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_gligen_v1_histogramfiltering(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [1, 2, 3, 5]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images_histogrammatching", annofile, sid))

        if syn_ver == "blendeddiff_v2_histogrammatching":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v2_histogrammatching(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "diffinpainting_v2_histogrammatching":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_diffinpainting_v2_histogrammatching(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "gligen_v2_histogrammatching":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_gligen_v2_histogrammatching(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [25]:
                for shot in [5, 10, 15, 25]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images_histogrammatching", annofile, sid))

        if syn_ver == "blendeddiff_v3_histogrammatching_1real_4syn":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v3_histogrammatching_1real_4syn(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "diffinpainting_v3_histogrammatching_1real_4syn":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_diffinpainting_v3_histogrammatching_1real_4syn(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "gligen_v3_histogrammatching_1real_4syn":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_gligen_v3_histogrammatching_1real_4syn(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [1, 2, 3, 5]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images_histogrammatching", annofile, sid))

        if syn_ver == "blendeddiff_v4_histogrammatching_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v4_histogrammatching_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "diffinpainting_v4_histogrammatching_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_diffinpainting_v4_histogrammatching_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "gligen_v4_histogrammatching_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_gligen_v4_histogrammatching_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [1, 2, 3, 5]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images_histogrammatching", annofile, sid))

        if syn_ver == "datazoo_v5_histogrammatching_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_datazoo_v5_histogrammatching_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [1, 2, 3, 5]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images", annofile, sid))

        if syn_ver == "datazoo_v5_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_datazoo_v5_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [5, 10, 15, 25]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images_bg_syn", annofile, sid))

        if syn_ver == "blendeddiff_v5_background_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v5_background_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "diffinpainting_v5_background_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_diffinpainting_v5_background_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "gligen_v5_background_1real_4syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_gligen_v5_background_1real_4syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )


        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [2, 4, 6, 10]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images_bg_syn", annofile, sid))

        if syn_ver == "blendeddiff_v6_background_1real_1syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v6_background_1real_1syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "diffinpainting_v6_background_1real_1syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_diffinpainting_v6_background_1real_1syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )
        elif syn_ver == "gligen_v6_background_1real_1syn_noleak":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_gligen_v6_background_1real_1syn_noleak(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

        METASPLITS = []
        for prefix in ["novel"]:
            for sid in [1]:
                for shot in [2, 4, 6, 10]:
                    name = "camo_{}_train_{}{}_{}shot".format(syn_ver, prefix, sid, shot)
                    annofile = format_template.format(sid) # will be ignored in the register_meta_camo() function
                    METASPLITS.append((name, "camo/images_bg_syn_noinstances", annofile, sid))

        if syn_ver == "blendeddiff_v7_background_1real_1syn_noleak_noinstances":
            for name, imgdir, annofile, sid in METASPLITS:
                register_meta_camo_blendeddiff_v7_background_1real_1syn_noleak_noinstances(
                    name,
                    _get_builtin_metadata("camo_{}_fewshot".format(syn_ver)),
                    os.path.join(root, imgdir),
                    os.path.join(root, annofile),
                    sid,
                )

# Register them all under "./datasets"
register_all_coco()
register_all_lvis()
register_all_pascal_voc()
register_all_camo()
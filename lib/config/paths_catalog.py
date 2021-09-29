"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "./datasets"

    DATASETS = {
        "tacos_train":{
            "video_dir": "tacos/",
            "ann_file": "tacos/annotations/train.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        "tacos_val":{
            "video_dir": "tacos/",
            "ann_file": "tacos/annotations/val.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        "tacos_test":{
            "video_dir": "tacos/",
            "ann_file": "tacos/annotations/test.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        ###########################################################################
        "activitynet_train":{
            "video_dir": "activitynet1.3/",
            "ann_file": "activitynet1.3/annotations/train.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        "activitynet_val":{
            "video_dir": "activitynet1.3/",
            "ann_file": "activitynet1.3/annotations/val.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        "activitynet_test":{
            "video_dir": "activitynet1.3/",
            "ann_file": "activitynet1.3/annotations/test.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        ###########################################################################
        "didemo_train":{
            "video_dir": "didemo/",
            "ann_file": "didemo/annotations/train.json",
            "feat_file": "didemo/features/didemo_5fps_vgg16_trimmed_original.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        "didemo_val":{
            "video_dir": "didemo/",
            "ann_file": "didemo/annotations/val.json",
            "feat_file": "didemo/features/didemo_5fps_vgg16_trimmed_original.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
        "didemo_test":{
            "video_dir": "didemo/",
            "ann_file": "didemo/annotations/test.json",
            "feat_file": "didemo/features/didemo_5fps_vgg16_trimmed_original.hdf5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
        },
    }


    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["video_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
            tokenizer_folder=os.path.join(data_dir, attrs["tokenizer_folder"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "didemo" in name:
            return dict(
                factory = "DidemoDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))

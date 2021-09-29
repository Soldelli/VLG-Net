import argparse
import os

import torch

from lib.config import get_cfg_defaults
from lib.data import make_data_loader
from lib.engine.inference import inference
from lib.modeling import build_model
from lib.utils.comm import synchronize, get_rank
from lib.utils.logger import setup_logger
from collections import OrderedDict

def count_parameters(model):
    train_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        train_params += parameter.numel()
    print(f"Total Trainable Params: {train_params}")


def main():
    parser = argparse.ArgumentParser(description="VLG")
    parser.add_argument(
        "--config-file",
        default="configs/activitynet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if not torch.cuda.is_available() and cfg.MODEL.DEVICE=='cuda':
        cfg.MODEL.DEVICE = 'cpu'    
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("vlg", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    count_parameters(model)

    output_dir = cfg.OUTPUT_DIR

    ############## load best model ##############
    best_checkpoint = f"{output_dir}/model_best_epoch.pth"
    if os.path.isfile(best_checkpoint):
        state_dict = torch.load(best_checkpoint, map_location=torch.device(cfg.MODEL.DEVICE))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
                name=dataset_name,
            )
            synchronize()
        

if __name__ == "__main__":
    main()

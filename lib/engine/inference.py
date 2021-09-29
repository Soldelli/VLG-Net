import logging
import time
import os
from tqdm import tqdm

import torch

from lib.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for batch in data_loader:
        batches, targets, idxs = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            batches = batches.to(device)
            output = model(batches.feats, batches.queries, batches.wordlens, batches.syntactic_dep)
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()

            # Move tensors to cpu
            output_cpu = []
            if len(idxs) == 1:
                output_cpu = [[o.to(cpu_device) for o in output]]
            else:
                for i in range(len(idxs)):
                    output_cpu.append([o[i].to(cpu_device) for o in output])


        results_dict.update(
            {img_id: result for img_id, result in zip(idxs, output_cpu)}
        )
    return results_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    idxs = list(sorted(predictions.keys()))
    if len(idxs) != idxs[-1] + 1:
        logger = logging.getLogger("vlg.inference")
        logger.warning(
            "Number of samples that were gathered from multiple processes is not "
            "a contiguous set. Some samples might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in idxs]
    return predictions

def inference(
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        name,
        device="cuda",
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("vlg.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / inference per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    iou_metrics = None
    if 'didemo' in name:
        iou_metrics = (0.5,0.7,1.0)
    elif 'activitynet' in name:
        iou_metrics = (0.5,0.7)
    elif 'tacos' in name:
        iou_metrics = (0.1,0.3,0.5)
    else:
        raise ValueError ('Unknown dataset. ')
    
    return evaluate(dataset=dataset, predictions=predictions, nms_thresh=nms_thresh, iou_metrics=iou_metrics) 

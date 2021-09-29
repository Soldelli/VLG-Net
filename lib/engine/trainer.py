import datetime
import logging
import os
import json
import time

import torch
import torch.distributed as dist

from lib.data import make_data_loader
from lib.utils.comm import is_main_process, get_world_size, synchronize
from lib.utils.metric_logger import MetricLogger
from lib.engine.inference import inference

def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            loss /= world_size
    return loss

def do_train(
    cfg,
    writer,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    dataset_name,
    data_loader_test=None,
):
    logger = logging.getLogger("vlg.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_recall = 0

    model.train()
    start_training_time = time.time()
    end = time.time()
    current_test_performance = None
    top_performance = torch.zeros((2,2))
    if 'activitynet' not in dataset_name:
        top_performance = torch.zeros((2,3))


    for epoch in range(arguments["epoch"], max_epoch + 1):
        max_iteration = len(data_loader)
        last_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch

        for iteration, (batches, targets, _) in enumerate(data_loader):
            iteration += 1
            data_time = time.time() - end

            batches = batches.to(device)
            targets = targets.to(device)
            
            def closure():
                optimizer.zero_grad()
                loss = model(batches.feats, 
                            batches.queries, 
                            batches.wordlens, 
                            batches.syntactic_dep, 
                            targets
                            )
                if writer is not None:
                    writer.add_scalar('train/loss', loss, epoch*max_iteration +iteration)
                if iteration % 20 == 0 or iteration == max_iteration:
                    meters.update(loss=reduce_loss(loss.detach()))
                loss.backward()
                return loss

            optimizer.step(closure)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iteration - iteration + last_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        scheduler.step()

        if epoch % checkpoint_period == 0:
            checkpointer.save(f"model_{epoch}e", **arguments)

        
        if data_loader_test is not None and test_period > 0 and \
            epoch % test_period == 0:
            synchronize()
            recall = inference(
                model,
                data_loader_test,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
                name=dataset_name,
            )
            synchronize()
            model.train()
            current_test_performance = recall

            if writer is not None and is_main_process():
                _write_to_tensorboard('test', dataset_name, recall, epoch, writer)

        if data_loader_val is not None and test_period > 0 and \
            epoch % test_period == 0:
            synchronize()
            recall = inference(
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.VAL,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
                name=dataset_name,
            )

            # Save current model if cyrrent val performance are higher than preivous validation performance
            if is_main_process():
                top_performance = save_if_best_model(cfg, epoch, model, dataset_name, 
                                top_performance, recall, current_test_performance)
                
            synchronize()
            model.train()

            if writer is not None and is_main_process():
                _write_to_tensorboard('val', dataset_name, recall, epoch, writer)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
    

def save_if_best_model(cfg, epoch, model, dataset_name, top_performance, val_recall, test_recall):
    '''
        Evaluate if current validation recall are higher than best recorder performance.
        Dump model and performance in files.
        Update top performance. 
    '''
    idx = 0
    if 'activitynet' not in dataset_name:
        idx = 1
    val_metric = (val_recall.mean(dim=0)[idx:] * torch.tensor([1,0.25])).sum().item()
    top_metric = (top_performance.mean(dim=0)[idx:] * torch.tensor([1,0.25])).sum().item()
    condition =  val_metric > top_metric

    # save
    if condition:
        top_performance = val_recall
        torch.save(model.state_dict(), cfg.OUTPUT_DIR +"/model_best_epoch.pth")
        _save_best_performance(dataset_name, test_recall, epoch, cfg)

    return top_performance

def _write_to_tensorboard(split, dataset_name, recall, epoch, writer):
    if 'didemo' in dataset_name:
        writer.add_scalar(f'{split}/r@1_IoU=0.5', recall[0][0].item(), epoch)
        writer.add_scalar(f'{split}/r@1_IoU=0.7', recall[0][1].item(), epoch)
        writer.add_scalar(f'{split}/r@1_IoU=1.0', recall[0][2].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=0.5', recall[1][0].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=0.7', recall[1][1].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=1.0', recall[1][2].item(), epoch)
    elif 'activitynet' in dataset_name:
        writer.add_scalar(f'{split}/r@1_IoU=0.5', recall[0][0].item(), epoch)
        writer.add_scalar(f'{split}/r@1_IoU=0.7', recall[0][1].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=0.5', recall[1][0].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=0.7', recall[1][1].item(), epoch)

    elif 'tacos' in dataset_name:        
        writer.add_scalar(f'{split}/r@1_IoU=0.1', recall[0][0].item(), epoch)
        writer.add_scalar(f'{split}/r@1_IoU=0.3', recall[0][1].item(), epoch)
        writer.add_scalar(f'{split}/r@1_IoU=0.5', recall[0][2].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=0.1', recall[1][0].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=0.3', recall[1][1].item(), epoch)
        writer.add_scalar(f'{split}/r@5_IoU=0.5', recall[1][2].item(), epoch)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def _save_best_performance(dataset_name, recall, epoch, cfg):
    dump_dict = {}
    if 'didemo' in dataset_name:
        dump_dict['r@1_IoU=0.5'] = recall[0][0].item()
        dump_dict['r@1_IoU=0.7'] = recall[0][1].item()
        dump_dict['r@1_IoU=1.0'] = recall[0][2].item()
        dump_dict['r@5_IoU=0.3'] = recall[1][0].item()
        dump_dict['r@5_IoU=0.5'] = recall[1][1].item()
        dump_dict['r@5_IoU=1.0'] = recall[1][2].item()
    elif 'activitynet' in dataset_name:
        dump_dict['r@1_IoU=0.5'] = recall[0][0].item()
        dump_dict['r@1_IoU=0.7'] = recall[0][1].item()
        dump_dict['r@5_IoU=0.3'] = recall[1][0].item()
        dump_dict['r@5_IoU=0.5'] = recall[1][1].item()

    elif 'tacos' in dataset_name:
        dump_dict['r@1_IoU=0.1'] = recall[0][0].item()
        dump_dict['r@1_IoU=0.3'] = recall[0][1].item()
        dump_dict['r@1_IoU=0.5'] = recall[0][2].item()
        dump_dict['r@5_IoU=0.1'] = recall[1][0].item()
        dump_dict['r@5_IoU=0.3'] = recall[1][1].item()
        dump_dict['r@5_IoU=0.5'] = recall[1][2].item()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

        
    dump_dict['epoch'] = epoch
    json.dump(dump_dict, open(cfg.OUTPUT_DIR +"/model_best_performance.json",'w'))



from terminaltables import AsciiTable
from tqdm import tqdm
import logging
import torch

from lib.data import datasets
from lib.data.datasets.utils import iou, score2d_to_moments_scores
from time import time
from math import ceil
import numpy as np

import joblib
from joblib import Parallel, delayed

def nms(moments, scores, topk, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True

    return moments[~suppressed]


def _eval_parallel(idx, step, predictions, durations, moments_list, num_recall_metrics, 
                    num_iou_metrics, recall_metrics, iou_metrics, num_clips, nms_thresh): 
    out = []
    for idx, score2d in zip(range(0,step), predictions):
        recall_x_iou_idx = torch.zeros(num_recall_metrics, num_iou_metrics)
        duration = durations[idx] 
        moment = moments_list[idx] 

        # Compute moment candidates and their scores
        candidates, scores = [], []
        for i, n in enumerate(num_clips):
            c, s = score2d_to_moments_scores(score2d[i], n, duration)
            candidates.append(c)
            scores.append(s)
        candidates = torch.cat(candidates)
        scores = torch.cat(scores)

        # Sort and compute performance
        moments = nms(candidates, scores, topk=recall_metrics[-1], thresh=nms_thresh)
        for i, r in enumerate(recall_metrics):
            mious = iou(moments[:r], moment)
            bools = mious[:,None].expand(r, num_iou_metrics) >= iou_metrics
            recall_x_iou_idx[i] += bools.any(dim=0)
        out.append(recall_x_iou_idx)
    return torch.stack(out,dim=0).sum(dim=0).numpy()


def evaluate(dataset, predictions, nms_thresh, recall_metrics=(1,5), iou_metrics=(0.1,0.3,0.5,0.7)):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """

    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("vlg.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    table = [['Rank@{},mIoU@{}'.format(i,j) \
        for i in recall_metrics for j in iou_metrics]]
    
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    num_clips = [p.shape[-1] for p in predictions[0]]

    num_cpu = joblib.cpu_count()
    num_predictions = len(predictions)
    step = ceil(num_predictions / num_cpu)
    
    predictions_  = [[predictions[i*step +j] for j in range(min(step, len(predictions)- i*step))] for i in range(num_cpu)]
    durations_    = [[dataset.get_duration(i*step +j) for j in range(min(step, len(predictions)- i*step))] for i in range(num_cpu)]
    moments_list_ = [[dataset.get_moment(i*step +j) for j in range(min(step, len(predictions)- i*step))] for i in range(num_cpu)]

    results = Parallel(n_jobs=num_cpu)(delayed(_eval_parallel)(
                        idx=idx, step=step, predictions=predictions_[idx], 
                        durations=durations_[idx], moments_list=moments_list_[idx], 
                        num_recall_metrics=num_recall_metrics, num_iou_metrics=num_iou_metrics, 
                        recall_metrics=recall_metrics, iou_metrics=iou_metrics,  
                        num_clips=num_clips,nms_thresh=nms_thresh) for idx in range(num_cpu))

    recall_x_iou = np.asarray(results).sum(axis=0)

    logger.info('{} is recall shape, should be [num_recall_metrics, num_iou_metrics]'.format(recall_x_iou.shape))
    recall_x_iou /= num_predictions
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)

    return torch.tensor(recall_x_iou)
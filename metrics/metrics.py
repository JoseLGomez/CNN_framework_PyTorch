import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import sys

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


# def evaluate(predictions, gts, num_classes):
#     hist = np.zeros((num_classes, num_classes))
#     for lp, lt in zip(predictions, gts):
#         hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
#     # axis 0: gt, axis 1: prediction
#     acc = np.diag(hist).sum() / hist.sum()
#     acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.000001)
#     acc_cls = np.nanmean(acc_cls)
#     iu = np.diag(hist) / ((hist.sum(axis=1) + 0.000001) + hist.sum(axis=0) - np.diag(hist))
#     mean_iu = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     return acc, acc_cls, mean_iu, fwavacc

#
# def evaluate(predictions, gts, num_classes):
#     hist = np.zeros((num_classes, num_classes))
#     for lp, lt in zip(predictions, gts):
#         hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
#     # axis 0: gt, axis 1: prediction
#     acc = np.diag(hist).sum() / hist.sum()
#     acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.000001)
#     acc_cls = np.nanmean(acc_cls)
#     iu = np.diag(hist) / ((hist.sum(axis=1) + 0.000001) + hist.sum(axis=0) - np.diag(hist))
#     mean_iu = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     return acc, acc_cls, mean_iu, fwavacc


def compute_stats(inputs,targets,nLabels,invalid_label):
    inputs_list = np.asarray(np.asarray(inputs).flatten())
    targets_list = np.asarray(np.asarray(targets).flatten())

    shape_inputs = np.shape(inputs_list)
    shape_targets = np.shape(targets_list)

    if shape_inputs != shape_targets:
        sys.exit('Error computing mIoU, inputs and targets shapes must be the same: ' + str(shape_inputs) + '==' + str(
            shape_targets))

    TP_list = np.zeros(nLabels)
    FP_list = np.zeros(nLabels)
    FN_list = np.zeros(nLabels)
    for l in range(nLabels):
        TP = float(np.sum(np.logical_and((targets_list == l),(inputs_list == l))))
        FP = float(np.sum(np.logical_and(np.logical_and((targets_list != l),(targets_list != invalid_label)),(inputs_list == l))))
        FN = float(np.sum(np.logical_and((targets_list == l),(inputs_list != l))))
        TP_list[l] = TP
        FP_list[l] = FP
        FN_list[l] = FN
    return TP_list,FP_list,FN_list


def compute_mIoU(TP_list,FP_list,FN_list):

    #mIoU_list = map(operator.div,TP_list,map(operator.add,TP_list,map(operator.add,FP_list,FN_list)))

    mIoU_list = np.zeros_like(TP_list)
    for i in range(len(mIoU_list)):
        if (TP_list[i] + FP_list[i] + FN_list[i]) == 0:
            mIoU_list[i] = 0.0
        else:
            mIoU_list[i] =  (TP_list[i]) / (TP_list[i] + FP_list[i] + FN_list[i])

    return mIoU_list
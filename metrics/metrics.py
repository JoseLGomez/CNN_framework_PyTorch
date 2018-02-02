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


def compute_stats(inputs,targets,nLabels,invalid_label):
    inputs_list = np.asarray(np.asarray(inputs).flatten())
    targets_list = np.asarray(np.asarray(targets).flatten())

    shape_inputs = np.shape(inputs_list)
    shape_targets = np.shape(targets_list)

    if shape_inputs != shape_targets:
        sys.exit('Error computing mIoU, inputs and targets shapes must be the same: ' + str(shape_inputs) + '==' + str(
            shape_targets))

    TP_list = np.zeros(nLabels)
    TN_list = np.zeros(nLabels)
    FP_list = np.zeros(nLabels)
    FN_list = np.zeros(nLabels)
    for l in range(nLabels):
        TP = float(np.sum(np.logical_and((targets_list == l),(inputs_list == l))))
        TN = float(np.sum(np.logical_and(np.logical_and((targets_list != l),(targets_list != invalid_label)), (inputs_list != l))))
        FP = float(np.sum(np.logical_and(np.logical_and((targets_list != l),(targets_list != invalid_label)),(inputs_list == l))))
        FN = float(np.sum(np.logical_and((targets_list == l),(inputs_list != l))))
        TP_list[l] = TP
        TN_list[l] = TN
        FP_list[l] = FP
        FN_list[l] = FN
    return TP_list,TN_list,FP_list,FN_list


def compute_mIoU(TP_list,FP_list,FN_list):

    mIoU_list = np.zeros_like(TP_list)
    for i in range(len(mIoU_list)):
        if (TP_list[i] + FP_list[i] + FN_list[i]) == 0:
            mIoU_list[i] = 0.0
        else:
            mIoU_list[i] =  (TP_list[i]) / (TP_list[i] + FP_list[i] + FN_list[i])

    return mIoU_list

def compute_precision(TP_list,FP_list):

    precision_list = np.zeros_like(TP_list)
    for i in range(len(precision_list)):
        if (TP_list[i] + FP_list[i]) == 0:
            precision_list[i] = 0.0
        else:
            precision_list[i] =  (TP_list[i]) / (TP_list[i] + FP_list[i])

    return precision_list

def compute_recall(TP_list,FN_list):

    recall_list = np.zeros_like(TP_list)
    for i in range(len(recall_list)):
        if (TP_list[i] + FN_list[i]) == 0:
            recall_list[i] = 0.0
        else:
            recall_list[i] =  (TP_list[i]) / (TP_list[i] + FN_list[i])

    return recall_list

def compute_accuracy(TP_list,TN_list, FP_list,FN_list):

    accuracy_list = np.zeros_like(TP_list)
    for i in range(len(accuracy_list)):
        if (TP_list[i] + TN_list[i] + FP_list[i] + FN_list[i]) == 0:
            accuracy_list[i] = 0.0
        else:
            accuracy_list[i] =  (TP_list[i] + TN_list[i]) / (TP_list[i] + TN_list[i] + FP_list[i] + FN_list[i])

    return accuracy_list

def compute_accuracy_segmentation(TP_list, FN_list):

    accuracy_list = np.zeros_like(TP_list)
    for i in range(len(accuracy_list)):
        if (TP_list[i] + FN_list[i]) == 0:
            accuracy_list[i] = 0.0
        else:
            accuracy_list[i] = (TP_list[i]) / (TP_list[i] + FN_list[i])

    return accuracy_list

def compute_f1score(TP_list,FP_list,FN_list):

    recall_list = compute_precision(TP_list, FN_list)
    precision_list = compute_precision(TP_list, FP_list)

    f1_score_list = np.zeros_like(TP_list)
    for i in range(len(f1_score_list)):
        if (precision_list[i] + recall_list[i]) == 0:
            f1_score_list[i] = 0.0
        else:
            f1_score_list[i] =  2.0 * (recall_list[i] + precision_list[i]) / (recall_list[i] + precision_list[i])

    return f1_score_list

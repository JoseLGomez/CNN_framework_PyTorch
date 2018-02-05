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

def extract_stats_from_confm(confm_list):
    TP_list = np.diag(confm_list)
    FP_list = confm_list.sum(axis=0) - TP_list                  # predictions on columns
    FN_list = confm_list.sum(axis=1) - TP_list                  # targets on rows
    TN_list = confm_list.sum() - TP_list - FP_list - FN_list

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
            f1_score_list[i] =  2.0 * (recall_list[i] * precision_list[i]) / (recall_list[i] + precision_list[i])

    return f1_score_list

def compute_confusion_matrix(inputs,targets,nLabels,invalid_label):

    inputs_list = np.asarray(np.asarray(inputs).flatten())
    targets_list = np.asarray(np.asarray(targets).flatten())

    mask = (targets_list!=invalid_label)

    conf_m = np.bincount(
        nLabels * targets_list[mask].astype(int) +
        inputs_list[mask], minlength=nLabels ** 2).reshape(nLabels, nLabels)

    # conf_m = np.zeros((nLabels,nLabels))
    #
    # for i in range(nLabels): # gt
    #     for j in range(nLabels): # predictions
    #         if np.sum(targets_list == i) == 0:
    #             conf_m[i, j] = 0
    #         else:
    #             conf_m[i,j] = float(np.sum(np.logical_and((targets_list == i),(inputs_list == j))))/np.sum(targets_list == i)

    return conf_m

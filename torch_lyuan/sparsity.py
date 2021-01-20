#----------------description----------------#
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-18 15:31:19
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-11-09 19:11:20
# FilePath     : /speech-to-text-wavenet/torch_lyuan/sparsity.py
# Description  :
#-------------------------------------------#
import os
import numpy as np

import torch
import torch.nn.functional as F
import sys
import config_train as cfg
import math
import time
from fast_pytorch_kmeans import KMeans

import sklearn.cluster._kmeans

# from sklearn.cluster import k_means
# from kmeans_pytorch import kmeans
import scipy.sparse

from itertools import combinations, permutations
from collections import defaultdict
from random import uniform
from math import sqrt



#----------------description----------------#
# description: prune the input model
# param {*} model
# param {*} sparse_mode
# return {*} pruned_model
#-------------------------------------------#


def pruning(model, sparse_mode='dense'):
    if sparse_mode == 'dense':
        return model

    elif sparse_mode == 'thre_pruning':
        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            # raw_w.topk()
            zero = torch.zeros_like(raw_w)
            if name.split(".")[-2] != "bn" \
                and name.split(".")[-2] != "bn2" \
                and name.split(".")[-2] != "bn3" \
                and name.split(".")[-1] != "bias":
                p_w = torch.where(abs(raw_w) < cfg.pruning_thre, zero, raw_w)
                zero_cnt += torch.nonzero(p_w).size()[0]
                all_cnt += torch.nonzero(raw_w).size()[0]
                a[name] = p_w
            else:
                a[name] = raw_w
        model.load_state_dict(a)

    elif sparse_mode == 'sparse_pruning':
        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)
            zero_num = int(w_num * cfg.sparsity)
            if name.split(".")[-2] != "bn" \
                and name.split(".")[-2] != "bn2" \
                and name.split(".")[-2] != "bn3" \
                and name.split(".")[-1] != "bias":
                value, _ = torch.topk(raw_w.abs().flatten(), w_num - zero_num)
                thre = abs(value[-1])
                zero = torch.zeros_like(raw_w)
                p_w = torch.where(abs(raw_w) < thre, zero, raw_w)

                zero_cnt += torch.nonzero(p_w).size()[0]
                all_cnt += torch.nonzero(raw_w).size()[0]
                a[name] = p_w
            else:
                a[name] = raw_w

        model.load_state_dict(a)

    elif sparse_mode == 'pattern_pruning':
        name_list = list()
        para_list = list()

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)

            # apply the patterns
            # mask = torch.tensor(cfg.pattern_mask[name])
            mask = cfg.pattern_mask[name].clone().detach()
            p_w = raw_w * mask
            a[name] = p_w
        model.load_state_dict(a)

    elif sparse_mode == 'coo_pruning':
        name_list = list()
        para_list = list()
        pattern_shape = cfg.coo_shape
        coo_nnz = cfg.coo_nnz

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)

            # apply the patterns
            mask = torch.zeros_like(raw_w)
            if name.split(".")[-2] != "bn" \
                and name.split(".")[-2] != "bn2" \
                and name.split(".")[-2] != "bn3" \
                and name.split(".")[-1] != "bias":
                # print(name, raw_w.size(), pattern_shape)
                if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                    for k in range(raw_w.size(2)):
                        assert raw_w.size(
                            0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                            assert raw_w.size(
                                1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                               oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k]
                                value, _ = torch.topk(
                                    part_w.abs().flatten(), coo_nnz)
                                thre = abs(value[-1])
                                zero = torch.zeros_like(part_w)
                                one = torch.ones_like(part_w)
                                part_mask = torch.where(
                                    abs(part_w) < thre, zero, one)
                                mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                     oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = part_mask

                    p_w = raw_w * mask
                    zero_cnt += torch.nonzero(p_w).size()[0]
                    all_cnt += torch.nonzero(raw_w).size()[0]
                    a[name] = p_w
                else:
                    a[name] = raw_w
            else:
                a[name] = raw_w

        model.load_state_dict(a)

    elif sparse_mode == 'ptcoo_pruning':
        name_list = list()
        para_list = list()
        pattern_shape = cfg.pattern_shape
        pt_nnz = cfg.pt_nnz
        coo_nnz = cfg.coo_nnz

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)

            # apply the patterns
            # mask = torch.tensor(cfg.pattern_mask[name])
            mask = cfg.pattern_mask[name].clone().detach()
            not_mask = torch.ones_like(cfg.pattern_mask[name]) - mask
            not_p_w = raw_w * not_mask

            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)

            # apply the patterns
            # mask = torch.zeros_like(raw_w)
            if name.split(".")[-2] != "bn" \
                and name.split(".")[-2] != "bn2" \
                and name.split(".")[-2] != "bn3" \
                and name.split(".")[-1] != "bias":
                # print(name, raw_w.size(), pattern_shape)
                if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                    for k in range(raw_w.size(2)):
                        assert raw_w.size(
                            0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                            assert raw_w.size(
                                1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                                not_part_w = not_p_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                                     oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k]
                                value, _ = torch.topk(
                                    not_part_w.abs().flatten(), coo_nnz)
                                thre = abs(value[-1])
                                zero = torch.zeros_like(not_part_w)
                                one = torch.ones_like(not_part_w)
                                part_mask = torch.where(
                                    abs(not_part_w) < thre, zero, one)
                                mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                     oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] += part_mask

                    p_w = raw_w * mask
                    zero_cnt += torch.nonzero(p_w).size()[0]
                    all_cnt += torch.nonzero(raw_w).size()[0]
                    a[name] = p_w
                else:
                    a[name] = raw_w
            else:
                a[name] = raw_w

        model.load_state_dict(a)

    elif sparse_mode == 'find_retrain':
        name_list = list()
        para_list = list()

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        if cfg.layer_or_model_wise == "l":
            for i, name in enumerate(name_list):
                raw_w = para_list[i]
                if name.split(".")[-2] != "bn" \
                    and name.split(".")[-2] != "bn2" \
                    and name.split(".")[-2] != "bn3" \
                    and name.split(".")[-1] != "bias":
                    if raw_w.size(0) == 128 and raw_w.size(1) == 40:
                        # raw_w_pad = torch.cat([raw_w, torch.zeros(raw_w.size(0), 4, raw_w.size(2)).cuda()], 1)
                        mask = apply_patterns(raw_w, cfg.fd_rtn_pattern_set[name], coo_num=cfg.coo_num)
                        mask =  mask[:, 0:raw_w.size(1), :]
                        p_w = raw_w * mask
                        a[name] = p_w
                    elif raw_w.size(0) == 128 and raw_w.size(1) == 128:
                        mask = apply_patterns(raw_w, cfg.fd_rtn_pattern_set[name], coo_num=cfg.coo_num)
                        p_w = raw_w * mask
                        a[name] = p_w
                    elif raw_w.size(0) == 28 and raw_w.size(1) == 128:
                        raw_w_pad = torch.cat([raw_w, torch.zeros(4, raw_w.size(1), raw_w.size(2)).cuda()], 0)
                        mask = apply_patterns(raw_w_pad, cfg.fd_rtn_pattern_set[name], coo_num=cfg.coo_num)
                        mask =  mask[0:raw_w.size(0), :, :]
                        p_w = raw_w * mask
                        a[name] = p_w
                    else:
                        a[name] = raw_w
                else:
                    a[name] = raw_w

        elif cfg.layer_or_model_wise == "m":
            for i, name in enumerate(name_list):
                raw_w = para_list[i]
                if name.split(".")[-2] != "bn" \
                    and name.split(".")[-2] != "bn2" \
                    and name.split(".")[-2] != "bn3" \
                    and name.split(".")[-1] != "bias":
                    if raw_w.size(0) == 128 and raw_w.size(1) == 128:
                        mask = apply_patterns(raw_w, cfg.fd_rtn_pattern_set['all'], coo_num=cfg.coo_num)
                        # ones = torch.ones_like(mask)
                        # zeros = torch.zeros_like(mask)
                        # print(mask.sum())
                        # print(torch.where(mask>1, ones, zeros).sum())
                        p_w = raw_w * mask
                        a[name] = p_w
                    else:
                        a[name] = raw_w
                else:
                    a[name] = raw_w
                    
        model.load_state_dict(a)

    elif sparse_mode == 'hcgs_pruning':
        name_list = list()
        para_list = list()

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)

            # apply the patterns
            # mask = torch.tensor(cfg.pattern_mask[name])
            mask = cfg.hcgs_mask[name].clone().detach()
            p_w = raw_w * mask
            a[name] = p_w
        model.load_state_dict(a)

    else:
        assert(False, "sparse mode does not exist")

    return model


def generate_pattern(pattern_num, pattern_shape, pattern_nnz):
    # generate the patterns
    patterns = torch.zeros([pattern_num, pattern_shape[0], pattern_shape[1]])
    for i in range(pattern_num):
        for j in range(pattern_nnz):
            random_row = np.random.randint(0, pattern_shape[0])
            random_col = np.random.randint(0, pattern_shape[1])
            # print(j, patterns[i, :, :])
            while patterns[i, random_row, random_col] == 1:
                random_row = np.random.randint(0, pattern_shape[0])
                random_col = np.random.randint(0, pattern_shape[1])
            patterns[i, random_row, random_col] = 1
        # print(patterns[i, :, :])
    return patterns


#----------------description----------------#
# description: use the given patterns to generate the masks of the input model
# param {*} model
# param {*} patterns
# return {*} patterns_mask
#-------------------------------------------#
def generate_pattern_mask(model, patterns):
    name_list = list()
    para_list = list()
    patterns_mask = dict()
    pattern_shape = [patterns.size(1), patterns.size(2)]
    pattern_num = patterns.size(0)

    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    a = model.state_dict()
    for i, name in enumerate(name_list):
        raw_w = para_list[i]
        w_num = torch.nonzero(raw_w).size(0)

        mask = torch.zeros_like(raw_w)
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
            if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                for k in range(raw_w.size(2)):
                    assert raw_w.size(
                        0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                    for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                        assert raw_w.size(
                            1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                        for oc_p in range(raw_w.size(1) // pattern_shape[1]):

                            mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                 oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = cfg.patterns[np.random.randint(0, pattern_num), :, :]

                patterns_mask[name] = mask

            else:
                patterns_mask[name] = torch.ones_like(raw_w)
        else:
            patterns_mask[name] = torch.ones_like(raw_w)

    # pattern_test = find_pattern_layer(patterns_mask[name], pattern_shape)
    # print(pattern_test.values())
    # print(len(pattern_test.values()))
    # exit()
    return patterns_mask


#----------------description----------------#
# description: sparsity = 1- (reserve_num1 / p_num_x) * (reserve_num2 / block_shape[0])
# param {*} model
# param {*} patterns
# return {*} patterns_mask
#-------------------------------------------#
def generate_hcgs_mask(model, block_shape, reserve_num1, reserve_num2):
    name_list = list()
    para_list = list()
    patterns_mask = dict()

    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    a = model.state_dict()
    for i, name in enumerate(name_list):
        raw_w = para_list[i]
        w_num = torch.nonzero(raw_w).size(0)
        mask = torch.zeros_like(raw_w)
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
            if raw_w.size(0) % block_shape[0] == 0 and raw_w.size(1) % block_shape[1] == 0:
                p_num_x = raw_w.size(0) // block_shape[0]
                p_num_y = raw_w.size(1) // block_shape[1]
                for k in range(raw_w.size(2)):
                    for oc in range(raw_w.size(1)):
                        reserve1_list = list(combinations(range(p_num_x), reserve_num1))
                        reserve1_list_len = len(reserve1_list)
                        # print(np.random.randint(0, p_num_x))
                        reserve1 = reserve1_list[int(np.random.randint(0, reserve1_list_len))]
                        for r1 in reserve1:
                            reserve2_list = list(combinations(range(block_shape[0]), reserve_num2))
                            reserve2_list_len = len(reserve2_list)
                            reserve2 = reserve2_list[int(np.random.randint(0, reserve2_list_len))]
                            for r2 in reserve2:
                                mask[oc, r1*16 + r2, k] = 1

                patterns_mask[name] = mask

            else:
                patterns_mask[name] = torch.ones_like(raw_w)
        else:
            patterns_mask[name] = torch.ones_like(raw_w)

    # pattern_test = find_pattern_layer(patterns_mask[name], pattern_shape)
    # print(pattern_test.values())
    # print(len(pattern_test.values()))
    # exit()
    return patterns_mask


#----------------description----------------#
# description: generate patterns and return the mask of input model.
#               the pattern set in different layers is different.
# param {*} model
# param {*} pattern_num
# param {*} pattern_shape
# param {*} pattern_nnz
# return {*} patterns_mask
#-------------------------------------------#
def generate_pattern_mask_layerwise(model, pattern_num, pattern_shape, pattern_nnz):
    name_list = list()
    para_list = list()
    patterns_mask = dict()
    patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)

    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    a = model.state_dict()
    for i, name in enumerate(name_list):

        patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)
        raw_w = para_list[i]
        w_num = torch.nonzero(raw_w).size(0)

        mask = torch.zeros_like(raw_w)
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
            if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                for k in range(raw_w.size(2)):
                    assert raw_w.size(
                        0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                    for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                        assert raw_w.size(
                            1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                        for oc_p in range(raw_w.size(1) // pattern_shape[1]):

                            mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                 oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = patterns[np.random.randint(0, pattern_num), :, :]

                patterns_mask[name] = mask

            else:
                patterns_mask[name] = torch.ones_like(raw_w)
        else:
            patterns_mask[name] = torch.ones_like(raw_w)

    # pattern_test = find_pattern_layer(patterns_mask[name], pattern_shape)
    # print(pattern_test.values())
    # print(len(pattern_test.values()))
    # exit()
    return patterns_mask


#----------------description----------------#
# description: 1)prune the model and reserve top-nnz weight.
#               2)save the patterns of top-nnz.
# param {*} model
# param {*} pattern_num
# param {*} pattern_shape
# param {*} pattern_nnz
# param {*} if_pattern_prun
# return {*} model, patterns
#-------------------------------------------#
def find_pattern_certain_nnz_model(model, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun=False):

    # pattern_num = 16
    # pattern_shape = [16, 16]
    # pattern_nnz = 32
    sparsity = pattern_nnz / (pattern_shape[0] * pattern_shape[1])
    patterns = dict()

    name_list = list()
    para_list = list()

    for name, para in model.named_parameters():
        if not para.dim() == 1:
            name_list.append(name)
            para_list.append(para)
            print(name, para.size())

    a = model.state_dict()
    zero_cnt = 0
    all_cnt = 0
    for i, name in enumerate(name_list):
        raw_w = para_list[i]
        raw_w, patterns_layer = find_pattern_certain_nnz_layer(
            raw_w, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun)
        patterns = add_dict(patterns, patterns_layer)
        if if_pattern_prun:
            a[name] = raw_w.squeeze(2)
    model.load_state_dict(a)

    return model, patterns


#----------------description----------------#
# description: 1)prune one layer and reserve top-nnz weight.
#               2)save the patterns of top-nnz.
# param {*} raw_w
# param {*} pattern_num
# param {*} pattern_shape
# param {*} pattern_nnz
# param {*} if_pattern_prun
# return {*} pruned_w, patterns
#-------------------------------------------#
def find_pattern_certain_nnz_layer(raw_w, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun=False):
    patterns = dict()
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    if not raw_w.size(0) % pattern_shape[0] == 0 or not raw_w.size(1) % pattern_shape[1] == 0:
        f"Error shape{raw_w.shape()}"
    mask = torch.ones_like(raw_w)
    for k in range(raw_w.size(2)):
        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                               oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k]
                value, _ = torch.topk(part_w.abs().flatten(), pattern_nnz)

                # pruning
                thre = abs(value[-1])
                one = torch.ones_like(part_w)
                zero = torch.zeros_like(part_w)
                mask_p = torch.where(abs(part_w) < thre, zero, one)

                mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                     oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = mask_p

                # save the pattern
    patterns = find_pattern_layer(mask, pattern_shape)
    if if_pattern_prun:
        sorted_patterns = sorted(
            patterns.keys(), key=lambda item: patterns[item], reverse=True)
        selected_pattern_list = sorted_patterns[:pattern_num]
        raw_w = pattern_prun_certain_nnz_layer(
            raw_w, selected_pattern_list, pattern_shape)
    return raw_w, patterns


#----------------description----------------#
# description: prune one layer using given patterns
# param {*} raw_w
# param {*} selected_pattern_list
# param {*} pattern_shape
# return {*} pruned_w
#-------------------------------------------#
def pattern_prun_certain_nnz_layer(raw_w, selected_pattern_list, pattern_shape):
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    selected_pattern_list = [torch.from_numpy(np.fromstring(selected_pattern_list[i], dtype=np.float32)).cuda(
    ).reshape(pattern_shape) for i in range(len(selected_pattern_list))]
    for k in range(raw_w.size(2)):
        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                               oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k].abs()
                pattern_index = max(range(len(selected_pattern_list)), key=lambda i: torch.sum(
                    selected_pattern_list[i]*(part_w)))
                raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                      oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] *= selected_pattern_list[pattern_index]
    return raw_w


#----------------description----------------#
# description: count the patterns in the model using sliding windows (no overlap).
# param {*} model
# param {*} pattern_shape      e.g. [16, 16]
# return {dict} patterns
#-------------------------------------------#
def find_pattern_model(model, pattern_shape):

    patterns = dict()

    name_list = list()
    para_list = list()

    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    for i, name in enumerate(name_list):
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
            raw_w = para_list[i]
            new_patterns = find_pattern_layer(raw_w, pattern_shape)
            patterns = add_dict(patterns, new_patterns)

    return patterns


#----------------description----------------#
# description: count the patterns in one layer using sliding windows (no overlap).
# param {tensor} raw_w          dim() = 2 or 3   e.g. (128,128,7) or (512, 512)
# param {list} pattern_shape    e.g. [16, 16]
# return {dict} patterns
#-------------------------------------------#
def find_pattern_layer(raw_w, pattern_shape):

    patterns = dict()
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
        for k in range(raw_w.size(2)):
            for ic_p in range(int(raw_w.size(0) / pattern_shape[0])):
                for oc_p in range(int(raw_w.size(1) / pattern_shape[1])):
                    part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                   oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k]
                    zero = torch.zeros_like(part_w)
                    one = torch.ones_like(part_w)
                    pattern = torch.where(
                        part_w == 0, zero, one).cpu().numpy().tobytes()
                    # pattern.squeeze(dim=0)
                    if part_w.size(0) == pattern_shape[0] and part_w.size(1) == pattern_shape[1]:
                        if pattern not in patterns.keys():
                            patterns[pattern] = 1
                        else:
                            patterns[pattern] += 1
    return patterns


#----------------description----------------#
# description: addition of two dicts.
# param {dict} x
# param {dict} y
# return {dict} x+y
#-------------------------------------------#
def add_dict(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] += v
        else:
            y[k] = v
    return y


#----------------description----------------#
# description: calculate the sparsity of the input model.
# param {*} model
# return {float} sparsity
#-------------------------------------------#
def cal_sparsity(model):
    name_list = list()
    para_list = list()
    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    zero_cnt = 0
    all_cnt = 0
    for i, name in enumerate(name_list):
        w = para_list[i]
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
            if w.size(0) == 128 and w.size(1) == 128:
                zero_cnt += w.flatten().size()[0] - torch.nonzero(w).size()[0]
                all_cnt += w.flatten().size()[0]

    return zero_cnt/all_cnt


#----------------description----------------#
# description:
# param {*} raw_w
# param {*} pattern_num
# param {*} pattern_shape
# param {*} zero_threshold
# param {*} coo_threshold
# return {*}
#-------------------------------------------#
def find_pattern_by_similarity(raw_w, pattern_num, pattern_shape, sparsity, coo_threshold):
    print(f'raw_w:{raw_w.size()}, \n\
            pattern_num: {pattern_num},  \n\
            pattern_shape: {pattern_shape}, \n\
            sparsity        : {sparsity},         \n\
            coo_threshold    : {coo_threshold}')
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)

    w_num = torch.nonzero(raw_w).size(0)
    zero_num = int(w_num * sparsity)
    value, _ = torch.topk(raw_w.abs().flatten(), w_num - zero_num)
    zero_threshold = abs(value[-1])

    # stride = [16, 16]
    stride = [pattern_shape[0], pattern_shape[1]]
    p_num_x = (raw_w.size(0) - pattern_shape[0])//stride[0] + 1
    p_num_y = (raw_w.size(1) - pattern_shape[1])//stride[1] + 1
    # mask = torch.zeros_like(raw_w).cuda()

    one = torch.ones_like(raw_w)
    zero = torch.zeros_like(raw_w)
    mask = torch.where(abs(raw_w) <= zero_threshold, zero, one).cuda()

    # get pattern candidates
    pattern_candidates = list()
    idx_to_ijk = dict()
    idx = 0
    for k in range(raw_w.size(2)):
        for i in range(0, p_num_x):
            for j in range(0, p_num_y):
                idx_to_ijk[idx] = [i, j, k]
                pattern_candidate = mask[i*stride[0]: i*stride[0] +
                                         pattern_shape[0], j*stride[1]: j*stride[1] + pattern_shape[1], k]
                pattern_candidates.append(pattern_candidate)
                idx += 1

    # output score maps
    score_maps = list()
    pattern_match_num_dict = dict()
    pattern_coo_nnz_dict = dict()
    pattern_nnz_dict = dict()
    pattern_inner_nnz_dict = dict()
    print(len(pattern_candidates))
    pattern_candidates, pattern_sort_index = sort_pattern_candidates(
        pattern_candidates)

    remove_bitmap = torch.zeros((p_num_x, p_num_y, raw_w.size(2)))

    print("sorted: ", len(pattern_candidates))
    for p_cnt, p in enumerate(pattern_candidates):
        if p.sum() < coo_threshold/2:
            continue
        p = 1 - p
        p_sum = p.sum()
        nnz_num = 0
        p_idx = pattern_sort_index[p_cnt]

        p_i = idx_to_ijk[p_idx][0]
        p_j = idx_to_ijk[p_idx][1]
        p_k = idx_to_ijk[p_idx][2]

        # print(p_i, p_j, p_k)
        if remove_bitmap[p_i, p_j, p_k] == 0:
            score_map = torch.zeros((p_num_x, p_num_y, raw_w.size(2)))
            for k in range(raw_w.size(2)):
                for i in range(0, p_num_x):
                    for j in range(0, p_num_y):
                        if remove_bitmap[i, j, k] == 1:
                            score_map[i, j, k] = coo_threshold + 1
                        else:
                            score_map[i, j, k] = (
                                p * mask[i*stride[0]: i*stride[0] + pattern_shape[0], j*stride[1]: j*stride[1] + pattern_shape[1], k]).sum()
            score_max = score_map.max()
            # assert score_max <= p_sum, f"{score_max} {p} {p_sum}"

            # remove the candidate score match the score threshold
            zeros = torch.zeros_like(remove_bitmap)
            ones = torch.ones_like(remove_bitmap)
            remove_bitmap_add = torch.where(
                score_map <= coo_threshold, ones, zeros)
            remove_bitmap = torch.where(
                remove_bitmap_add >= 1, ones, remove_bitmap)

            p = 1-p
            match_num = remove_bitmap_add.sum()
            print(p_cnt, idx_to_ijk[p_idx], ",current_pattern_nnz:", int(p.sum()),
                  ",output_max:", int(score_max),
                  ",match_num:", int(match_num),
                  ",removed:", int(remove_bitmap.sum()))
            pattern_inner_nnz_dict[p.cpu().numpy().tobytes()] = p.sum()
            pattern_match_num_dict[p.cpu().numpy().tobytes()] = match_num
            pattern_coo_nnz_dict[p.cpu().numpy().tobytes()] = (
                score_map * remove_bitmap_add).sum()

            for k in range(raw_w.size(2)):
                for i in range(0, p_num_x):
                    for j in range(0, p_num_y):
                        if remove_bitmap_add[i, j, k] == 1:
                            nnz_num += mask[i*stride[0]: i*stride[0] + pattern_shape[0],
                                            j*stride[1]: j*stride[1] + pattern_shape[1], k].sum()

            pattern_nnz_dict[p.cpu().numpy().tobytes()] = nnz_num
        else:
            pass

        # # save more
    assert int(remove_bitmap.sum()) == len(pattern_candidates)
    # if int(remove_bitmap.sum()) == len(pattern_candidates):
    #     break
    print(len(pattern_match_num_dict))

    # collect top-pattern_num patterns
    if len(pattern_match_num_dict.items()) < pattern_num:
        pattern_num = len(pattern_match_num_dict)
    patterns = list()
    pattern_match_num = list()
    pattern_coo_nnz = list()
    pattern_nnz = list()
    pattern_inner_nnz = list()
    patterns = sorted(zip(pattern_match_num_dict.values(),
                          pattern_match_num_dict.keys()), reverse=True)
    # patterns = sorted(pattern_match_num_dict, key = lambda k: k[pattern_num])
    for p in patterns:
        # print(p)
        # print(p[0])
        p = p[1]
        pattern_match_num.append(pattern_match_num_dict[p])
        # print(pattern_match_num_dict[p], " ", end="")
        pattern_coo_nnz.append(pattern_coo_nnz_dict[p])
        pattern_nnz.append(pattern_nnz_dict[p])
        pattern_inner_nnz.append(pattern_inner_nnz_dict[p])
        # patterns[p] = score
        p = np.frombuffer(p, dtype=np.float32).reshape(pattern_shape)
        # print(p, score)

    # exit()
    return patterns, np.array(pattern_match_num), np.array(pattern_coo_nnz), np.array(pattern_nnz), np.array(pattern_inner_nnz)


#----------------description----------------#
# description:
# param {*} pattern_match_num_dict
# param {*} pattern_coo_nnz_dict
# param {*} pattern_nnz_dict
# return {*} pattern_num_memory_dict, pattern_num_coo_nnz_dict
#-------------------------------------------#
def pattern_curve_analyse(raw_w_shape, pattern_shape, patterns, pattern_match_num, pattern_coo_nnz, pattern_nnz, pattern_inner_nnz):

    submatrix_num = (raw_w_shape[0] // pattern_shape[0]) * \
        (raw_w_shape[1] // pattern_shape[1])
    pattern_num_memory_dict = dict()
    pattern_num_cal_num_dict = dict()
    pattern_num_coo_nnz_dict = dict()
    pattern_num_list = [1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512]
    for pattern_num in pattern_num_list:
        if pattern_num >= len(patterns) * 2:
            break
        pattern_bit_num = pattern_inner_nnz[:pattern_num].sum(
        ) * (math.log(pattern_shape[0], 2) + math.log(pattern_shape[1], 2))

        if pattern_num == 1:
            pattern_bit = 1
        else:
            pattern_bit = math.log(pattern_num, 2)

        pattern_idx_bit_num = pattern_bit \
            * pattern_match_num[:pattern_num].sum()
        # pattern_idx_bit_num = pattern_bit \
        #                     * submatrix_num
        coo_idx_num = pattern_coo_nnz[:pattern_num].sum() \
            + pattern_nnz[pattern_num:].sum()
        coo_idx_bit_num = (math.log(pattern_shape[0], 2) + math.log(pattern_shape[1], 2)) \
            * coo_idx_num
        memory_cost = pattern_idx_bit_num + coo_idx_bit_num + pattern_bit_num

        cal_num = (pattern_match_num[:pattern_num] * pattern_inner_nnz[:pattern_num]).sum() \
            + coo_idx_num

        pattern_num_memory_dict[pattern_num] = memory_cost     #
        pattern_num_cal_num_dict[pattern_num] = cal_num        # cal num
        pattern_num_coo_nnz_dict[pattern_num] = coo_idx_num    # left coo

    # print(pattern_num_memory_dict)
    return pattern_num_memory_dict, pattern_num_cal_num_dict, pattern_num_coo_nnz_dict


#----------------description----------------#
# description:
# param {*} pattern_candidates
# return {*}
#-------------------------------------------#
def sort_pattern_candidates(pattern_candidates):
    pattern_candidates_sorted = pattern_candidates.copy()
    pattern_sort_index = sorted(range(len(
        pattern_candidates)), key=lambda k: pattern_candidates[k].sum(), reverse=False)
    # print(pattern_sort_index)
    pattern_candidates_sorted = [
        pattern_candidates_sorted[i] for i in pattern_sort_index]
    # for i in range(len(pattern_candidates)-1):
    #     for j in range(len(pattern_candidates)-1-i):
    #         if pattern_candidates[j].sum() > pattern_candidates[j+1].sum():
    #             pattern_candidates[j], pattern_candidates[j+1] = pattern_candidates[j+1], pattern_candidates[j]

    return pattern_candidates_sorted, pattern_sort_index


# #----------------description----------------#
# # description: find pattern by similarity
# #                step 1: generate pattern candidates
# #                step 2: calculate the output score mask (remove patterns)
# #                step 3: return top-k patterns
# # param {*} raw_w
# # param {*} pattern_num
# # param {*} pattern_shape
# # param {*} zero_threshold
# # param {*} score_threshold
# # return {*}
# #-------------------------------------------#
# def find_pattern_envelope_by_similarity(raw_w, pattern_num, pattern_shape, zero_threshold, score_threshold):
#     if raw_w.dim() == 2:
#         raw_w = raw_w.unsqueeze(2)

#     # mask = torch.zeros_like(raw_w).cuda()

#     one = torch.ones_like(raw_w)
#     zero = torch.zeros_like(raw_w)
#     mask = torch.where(abs(raw_w) <= zero_threshold, zero, one).cuda()

#     # get pattern candidates
#     pattern_candidates = list()
#     idx_to_ijk = dict()
#     idx = 0
#     for k in range(raw_w.size(2)):
#         for i in range(raw_w.size(0) - pattern_shape[0] +1):
#             for j in range(raw_w.size(1) - pattern_shape[1] +1):
#                 idx_to_ijk[idx] = [i, j, k]
#                 # part_w = raw_w[i: i + pattern_shape[0],
#                 #                 j: j + pattern_shape[1], k]
#                 # one = torch.ones_like(part_w)
#                 # zero = torch.zeros_like(part_w)
#                 # pattern_candidate = torch.where(abs(part_w) <= zero_threshold, zero, one).cuda()
#                 pattern_candidate = mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k]
#                 # mask[i: i + pattern_shape[0],
#                 #     j: j + pattern_shape[1], k] = pattern_candidate

#                 pattern_candidates.append(pattern_candidate)
#                 idx += 1


#     # output score maps
#     score_maps = list()
#     pattern_match_num_dict = dict()
#     pattern_match_nnz_dict = dict()
#     print(len(pattern_candidates))
#     pattern_candidates, pattern_sort_index = sort_pattern_candidates(pattern_candidates)
#     remove_bitmap = torch.zeros((raw_w.size(0) - pattern_shape[0] +1, raw_w.size(1) - pattern_shape[1] +1, raw_w.size(2)))

#     print("sorted: ", len(pattern_candidates))
#     for p_num, p in enumerate(pattern_candidates):
#         p_idx = pattern_sort_index[p_num]

#         p_i = idx_to_ijk[p_idx][0]
#         p_j = idx_to_ijk[p_idx][1]
#         p_k = idx_to_ijk[p_idx][2]

#         # print(p_i, p_j, p_k)
#         if remove_bitmap[p_i, p_j, p_k] == 0:
#             score_map = torch.zeros((raw_w.size(0) - pattern_shape[0] +1, raw_w.size(1) - pattern_shape[1] +1, raw_w.size(2)))

#             for k in range(raw_w.size(2)):
#                 for i in range(raw_w.size(0) - pattern_shape[0] +1):
#                     for j in range(raw_w.size(1) - pattern_shape[1] +1):
#                         if remove_bitmap[i, j, k] == 1:
#                             score_map[i, j, k] = 0
#                         else:
#                             score_map[i, j, k] = (p * mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k]).sum()

#                             # if i == p_i and j == p_j and k == p_k:
#                             #     # print(pattern_candidates[p_idx])
#                             #     print(p)
#                             #     print(mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k])
#                             #     print((p * mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k]).sum())

#             # score_maps.append(score_map)
#             score_max = score_map.max()
#             assert score_max <= p.sum(), f"{score_max} {p} {p.sum()}"

#             # remove the candidate score match the score threshold
#             zeros = torch.zeros_like(remove_bitmap)
#             ones = torch.ones_like(remove_bitmap)
#             # print(remove_bitmap)
#             # print(score_max, score_threshold)
#             remove_bitmap_add = torch.where(score_map >= abs(score_max-score_threshold), ones, zeros)
#             # print(torch.nonzero(remove_bitmap_add))
#             remove_bitmap = torch.where(remove_bitmap_add >= 1, ones, remove_bitmap)

#             # print(remove_bitmap)
#             match_num = remove_bitmap_add.sum()
#             print(p_num, idx_to_ijk[p_idx], ",current_pattern_nnz:", int(p.sum()),
#                                     ",output_max:", int(score_max),
#                                     ",score:", int(match_num),
#                                     ",removed:", int(remove_bitmap.sum()))
#             pattern_match_num_dict[p.cpu().numpy().tobytes()] = match_num
#         else:
#             pass

#         if len(pattern_match_num_dict.keys()) >= 200:
#             break

#     print(len(pattern_match_num_dict))
#     # if_save = False
#     # if if_save:


#     # collect top-pattern_num patterns
#     if len(pattern_match_num_dict.items()) < pattern_num:
#         pattern_num = len(pattern_match_num_dict)
#     patterns = dict()
#     pattern_match_num_dict_sorted = sorted(pattern_match_num_dict, key = lambda k: k[pattern_num])
#     for p in pattern_match_num_dict_sorted:
#         score = pattern_match_num_dict[p]
#         patterns[p] = score
#         p = np.frombuffer(p, dtype=np.float32).reshape(pattern_shape)
#         # print(p, score)

#     # exit()
#     return patterns, pattern_match_num_dict, pattern_match_nnz_dict


def generate_complete_pattern_set(pattern_shape, pattern_nnz):
    pattern_set = list()
    pattern_total_num = pattern_shape[0]*pattern_shape[1]
    pattern_set_len = comb_num(pattern_total_num, pattern_nnz)
    assert pattern_set_len <= 50000, f"Pattern candidate set too big! {pattern_set_len}"

    pattern_nnz_pos_list = list(combinations(
        range(pattern_total_num), pattern_nnz))
    for pattern_nnz_pos in pattern_nnz_pos_list:
        pattern = torch.zeros((pattern_shape[0], pattern_shape[1]))
        for nnz_idx in pattern_nnz_pos:
            ic = nnz_idx % pattern_shape[0]
            oc = nnz_idx // pattern_shape[0]
            pattern[ic, oc] = 1
        pattern_set.append(pattern)

    return pattern_set


def find_top_k_by_similarity(raw_w, pattern_candidates, stride, pattern_num):
    pattern_shape = [pattern_candidates[0].size(0), pattern_candidates[0].size(1)]
    p_num_x = (raw_w.size(0) - pattern_shape[0])//stride[0] + 1
    p_num_y = (raw_w.size(1) - pattern_shape[1])//stride[1] + 1
    pattern_score = dict()
    raw_w = torch.abs(raw_w)
    if raw_w.device.type == 'cpu':
        raw_w = raw_w.cuda()
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    # start_t = time.time()

    kernel_candidate = torch.zeros((len(pattern_candidates), raw_w.size(2), pattern_shape[0], pattern_shape[1])).cuda()
    for i in range(raw_w.size(2)):
        for p_i, p in enumerate(pattern_candidates):
            kernel_candidate[p_i, i, :, :] = pattern_candidates[p_i]

    # print(raw_w.unsqueeze(0).size())
    # print(kernel_candidate.size())
    raw_w = raw_w.permute(2, 0, 1)
    out = F.conv2d(raw_w.unsqueeze(0), kernel_candidate, stride=stride, padding=0)
    # print("out:", out.size())
    scores = out.sum(2).sum(2).squeeze(0)
    # print(scores)
    for i, score in enumerate(scores):
        pattern_score[pattern_candidates[i].cpu().numpy().tobytes()] = score

    # for i, p in enumerate(pattern_candidates):
    #     pattern_score[pattern_candidates[i].cpu().numpy().tobytes()] = i


    patterns = sorted(zip(pattern_score.values(),
                          pattern_score.keys()), reverse=True)[:pattern_num]
    pattern_set = [np.frombuffer(p[1], dtype=np.float32).reshape(
        pattern_shape) for p in patterns]
    pattern_set = [(torch.from_numpy(p)).cuda() for p in pattern_set]

    kernel = torch.zeros((len(pattern_set), 1, pattern_shape[0], pattern_shape[1])).cuda()
    for p_i, p in enumerate(pattern_set):
        kernel[p_i, 0, :, :] = pattern_set[p_i]

    return kernel

def find_top_k_by_kmeans(raw_w, pattern_num, pattern_shape, pattern_nnz, stride, kmeans_lib = 'sklearn', random=False):
    start_t = time.time()
    p_num_x = (raw_w.size(0) - pattern_shape[0])//stride[0] + 1
    p_num_y = (raw_w.size(1) - pattern_shape[1])//stride[1] + 1
    pattern_total_num = pattern_shape[0]*pattern_shape[1]
    pattern_set_len = comb_num(pattern_total_num, pattern_nnz)
    if pattern_set_len < pattern_num:
        pattern_num = pattern_set_len

    raw_w = torch.abs(raw_w)
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    if raw_w.device.type == 'cuda':
        raw_w = raw_w.cpu().detach().numpy()
        
    # print(f"find_top_k_by_kmeans 1.move to cpu cost===={time.time()-start_t} s. \
    #     pattern_shape:{raw_w.shape},pattern_num:{pattern_num},pattern_nnz:{pattern_nnz}")


    # # weight padding
    # min_size = max(pattern_shape) * 16
    # raw_w = zero_pad(raw_w, min_size,  w_type="array")

    # # rearrange the ic,oc by pe array_shape[16*16]
    # for k in range(raw_w.shape[2]):
    #     raw_w[:, :, k] = rearrange(raw_w[:, :, k], w_type="array")

    if random == False:
        pattern_candidates = list()
        for k in range(raw_w.shape[2]):
            for i in range(0, p_num_x):
                for j in range(0, p_num_y):
                    # print(raw_w.shape)
                    # print(k,i,j,p_num_x, p_num_y)
                    # print(pattern_num, pattern_shape, pattern_nnz, stride)
                    # print(i*stride[0], i*stride[0] + pattern_shape[0],
                                            #   j*stride[1], j*stride[1] + pattern_shape[1], k)
                    sub_matrix = np.abs(raw_w[i*stride[0]: i*stride[0] + pattern_shape[0],
                                            j*stride[1]: j*stride[1] + pattern_shape[1], k]).flatten()
                    # print("sub_matrix:", sub_matrix.shape)
                    index = sub_matrix.argsort()[-pattern_nnz:]
                    pattern_candidate = np.zeros_like(sub_matrix)
                    for t in index:
                        pattern_candidate[t] = 1
                    # value, _ = torch.topk(sub_matrix.abs().flatten(), pattern_nnz)
                    # zero_threshold = value[-1]

                    # ones = torch.ones_like(sub_matrix)
                    # zeros = torch.zeros_like(sub_matrix)
                    # pattern_candidate = torch.where(abs(sub_matrix) < zero_threshold, zeros, ones)
                    # print("pattern_candidate:", pattern_candidate.shape)

                    pattern_candidates.append(pattern_candidate)
        
        pattern_candidates = torch.tensor(np.array(pattern_candidates))
        # print(len(pattern_candidates))
        # print(pattern_candidates[0].shape)
        #     print(f"candidate_pattern length:{len(pattern_candidates)},pattern class{len(set(pattern_candidates))}")
        # print(f"find_top_k_by_kmeans 2.pattern candidate cost===={time.time()-start_t} s. \
        #     pattern_shape:{raw_w.shape},pattern_num:{pattern_num},pattern_nnz:{pattern_nnz}")
        # centers, _, _, n = k_means(X=pattern_candidates, n_clusters=pattern_num, return_n_iter=True, max_iter=3)
        if kmeans_lib == 'sklearn':
            clf = sklearn.cluster.KMeans(n_clusters=pattern_num)
            clf.fit(pattern_candidates)
            centers = clf.cluster_centers_
        elif kmeans_lib == 'fast':
            pattern_candidates = pattern_candidates.to(torch.float32)
            kmeans = KMeans(n_clusters=pattern_num, mode='euclidean', verbose=1)
            labels = kmeans.fit_predict(pattern_candidates)
            centers = kmeans.centroids
        # print(kmeans.centroids.size())
        # print(labels)
        # print(f"find_top_k_by_kmeans 3. clustering cost====={time.time()-start_t} s. \
        #     pattern_shape:{raw_w.shape},pattern_num:{pattern_num},pattern_nnz:{pattern_nnz}")
        # # clf.fit(pattern_candidates)  # 分组

        # centers = clf.cluster_centers_ # 两组数据点的中心点

    pattern_set = list()
    if random == False:
        for pattern in centers:
            if kmeans_lib == 'sklearn':
                pattern = torch.from_numpy(pattern)
            # print(pattern.size())
            index = pattern.sort()[1][-pattern_nnz:]
            pattern = torch.zeros_like(pattern)
            for i in index:
                pattern[i] = 1

            pattern_set.append(pattern.reshape(pattern_shape[0], pattern_shape[1]))
    else:
        pattern_set_tensor = generate_pattern(pattern_num, pattern_shape, pattern_nnz)
        for i in range(pattern_set_tensor.size(0)):
            pattern_set.append(pattern_set_tensor[i, :, :])

    kernel = torch.zeros((len(pattern_set), 1, pattern_shape[0], pattern_shape[1])).cuda()
    for p_i, p in enumerate(pattern_set):
        kernel[p_i, 0, :, :] = pattern_set[p_i]
    # print(f"find_top_k_by_kmeans 3. clustering cost====={time.time()-start_t} s. \
    #     pattern_shape:{raw_w.shape},pattern_num:{pattern_num},pattern_nnz:{pattern_nnz}")
    return kernel, True

def raw_w_list2raw_w_chunk(raw_w_list):
    assert raw_w_list[0].size(0) == raw_w_list[0].size(1)
    assert raw_w_list[0].size(0) == 128
    raw_w_chunk = raw_w_list[0]
    batch_list = list()
    batch_list.append(raw_w_list[0].size(2))
    for raw_w in raw_w_list[1:]:
        raw_w_chunk = torch.cat([raw_w_chunk, raw_w], 2)
        batch_list.append(raw_w.size(2))
    # print(raw_w_chunk.size())
    return raw_w_chunk, batch_list


def apply_patterns_chunk(raw_w_chunk, batch_list, kernel):
    # print(raw_w.size())
    raw_w = torch.abs(raw_w_chunk)
    start_t = time.time()
    pattern_shape = [kernel.size(2), kernel.size(3)]
    stride = (pattern_shape[0], pattern_shape[1])
    p_num_x = (raw_w.size(0) - pattern_shape[0]) // stride[0] + 1
    p_num_y = (raw_w.size(1) - pattern_shape[1]) // stride[1] + 1

    if raw_w.device.type == 'cpu':
        raw_w = raw_w.cuda()
    unsqueeze = False
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
        unsqueeze = True

    mask = torch.zeros_like(raw_w).cuda()
    raw_w = raw_w.permute(2, 0, 1)
    
    out = F.conv2d(raw_w.unsqueeze(1), kernel, stride=stride, padding=0)

    for i, batch in enumerate(batch_list):
        start = np.array(batch_list)[:i].sum()
        # print(start, batch, raw_w.size(0))
        idx = torch.argmax(out[start:start+batch], dim=1).squeeze(0)
        if idx.dim() == 2:
            idx = idx.unsqueeze(0)
        for k in range(batch):
            for i in range(0, p_num_x):
                for j in range(0, p_num_y):
                    mask[i*stride[0]: i*stride[0] + pattern_shape[0], j*stride[1]
                        : j*stride[1] + pattern_shape[1], start+k] = kernel[idx[k][i][j], 0, :, :]

    # print("apply one layer time==================", time.time() - start_t)
    if unsqueeze == True:
        mask = mask.squeeze(2)
    return mask

def mask_chunk2mask_list(mask_chunk, batch_list):
    mask_list = list()
    for i, batch in enumerate(batch_list):
        start = np.array(batch_list)[:i].sum()
        mask = mask_chunk[:, :, start:start+batch]
        mask_list.append(mask)
    return mask_list

def baseline_pruning(raw_w, pattern_shape, pattern_nnz):

    raw_w_a = torch.abs(raw_w)
    if raw_w_a.device.type == 'cpu':
        raw_w_a = raw_w_a.cuda()
    unsqueeze = False
    if raw_w_a.dim() == 2:
        raw_w_a = raw_w_a.unsqueeze(2)
        unsqueeze = True
    
    p_num_x = int(raw_w_a.size(0) / pattern_shape[0])
    p_num_y = int(raw_w_a.size(1) / pattern_shape[1])
    submatrix_num = p_num_y * p_num_x

    value, _ = torch.topk(
        raw_w_a.abs().flatten(), int(submatrix_num * pattern_nnz))
    thre = abs(value[-1])
    zeros = torch.zeros_like(raw_w_a)
    ones = torch.ones_like(raw_w_a)
    mask = torch.where(
        abs(raw_w_a) < thre, zeros, ones)
    
    if unsqueeze == True:
        mask = mask.squeeze(2)
    # exit()
    return mask

def apply_patterns(raw_w, kernel, coo_num=0, random=False):
    # print(raw_w.size())
    if random == True:
        raw_w_a = torch.randn(raw_w.size()).cuda()
        raw_w_a = torch.abs(raw_w_a)
    else:
        raw_w_a = torch.abs(raw_w)
    # pattern_set = [(torch.from_numpy(p)) for p in pattern_set]
    start_t = time.time()
    # pattern_shape = [pattern_set[0].size(0), pattern_set[0].size(1)]
    pattern_shape = [kernel.size(2), kernel.size(3)]
    stride = (pattern_shape[0], pattern_shape[1])
    # print(pattern_shape)
    p_num_x = (raw_w_a.size(0) - pattern_shape[0]) // stride[0] + 1
    p_num_y = (raw_w_a.size(1) - pattern_shape[1]) // stride[1] + 1

    if raw_w_a.device.type == 'cpu':
        raw_w_a = raw_w_a.cuda()
    unsqueeze = False
    if raw_w_a.dim() == 2:
        raw_w_a = raw_w_a.unsqueeze(2)
        unsqueeze = True

    # # weight padding
    # min_size = 8 * 16
    # raw_w_a = zero_pad(raw_w_a, min_size, "tensor")
    # # rearrange the arrangement of weight
    # for k in range(raw_w_a.shape[2]):
    #     raw_w_a[:, :, k] = rearrange(raw_w_a[:, :, k])


    mask = torch.zeros_like(raw_w_a).cuda()
    # 7,128,128
    raw_w_a = raw_w_a.permute(2, 0, 1)

    if raw_w_a.device.type == 'cpu':
        raw_w_a = raw_w_a.cuda()
    
    out = F.conv2d(raw_w_a.unsqueeze(1), kernel, stride=stride, padding=0)

    out_max = torch.max(out, dim=1)[0].unsqueeze(1).repeat(1,kernel.size(0),1,1)
    idx = torch.where(out >= out_max, torch.ones_like(out), torch.zeros_like(out))
    mask = torch.nn.functional.conv_transpose2d(idx, kernel, 
                bias=None, stride=stride, padding=0, output_padding=0, groups=1)
    mask = mask.squeeze(1)
    # print("apply one layer time==================", time.time() - start_t)

    # 128,128,7
    mask = mask.permute(1, 2, 0)

    w_num = raw_w_a.size(0) * raw_w_a.size(1) * raw_w_a.size(2)
    if coo_num != 0:
        mask_r = 1 - mask
        raw_w_a = raw_w_a.permute(1, 2, 0)
        coo_w = raw_w_a * mask_r

        value, _ = torch.topk(
            coo_w.abs().flatten(), int(coo_num * w_num / (pattern_shape[0]*pattern_shape[1])))
        thre = abs(value[-1])
        zero = torch.zeros_like(coo_w)
        one = torch.ones_like(coo_w)
        coo_mask = torch.where(
            abs(coo_w) < thre, zero, one)
        mask += coo_mask

    # # recover the arrangement of mask
    # for k in range(mask.shape[2]):
    #     mask[:, :, k] = recoverorder(mask[:, :, k])
    # # de zero_pad of mask. not finish yet

    if unsqueeze == True:
        mask = mask.squeeze(2)
    # exit()
    return mask


def rearrange(weight, pe_row=16, pe_col=16, w_type="tensor"):
    if w_type == "array":
        weight = np.concatenate(np.split(weight.reshape(weight.shape[0]//pe_col,-1),pe_row,1),0).T     
        weight = np.concatenate(np.split(weight.reshape(weight.shape[0]//pe_row,-1),pe_col,1),0).T
    elif w_type == "tensor":
        weight = torch.cat(torch.split(weight.reshape(weight.shape[0]//pe_row,-1),weight.shape[0],1),0).t()
        weight = torch.cat(torch.split(weight.reshape(weight.shape[0]//pe_row,-1),weight.shape[0],1),0).t()
    else:
        print("Weight type Error:", w_type)
    return weight

def recoverorder(weight, pe_row=16, pe_col=16, w_type="tensor"):
    # w_type: "tensor" or "array"
    if w_type == "array":
        weight = np.concatenate(np.split(weight,pe_col,0),1).reshape(weight.shape[0], -1).T
        weight = np.concatenate(np.split(weight,pe_row,0),1).reshape(weight.shape[0], -1).T
    elif w_type == "tensor":
        weight = torch.cat(torch.split(weight,weight.shape[0]//pe_col,0),1).reshape(weight.shape[0], -1).t()
        weight = torch.cat(torch.split(weight,weight.shape[0]//pe_row,0),1).reshape(weight.shape[0], -1).t()
    else:
        print("Weight type Error:", w_type)
    return weight

def zero_pad(x, min_size=128, w_type="array"):
    unsqueeze = False
    if len(x.shape)==2:
        if w_type == "array":
            x = np.expand_dims(x, 2)
        elif w_type == "tensor":
            x = x.unsqueeze(2)
        unsqueeze = True
    # w_type: "tensor" or "array"
    old_shape = list(x.shape)
    new_shape = list(x.shape)
    for dim in range(len(old_shape)-1):
        if old_shape[dim]%min_size:
            new_shape[dim] = ((old_shape[dim]//min_size)+1)*min_size
    if w_type == "array":
        new_x = np.zeros(new_shape)
        new_x[:x.shape[0], :x.shape[1], :] = x
    elif w_type == "tensor":
        new_x = torch.zeros(new_shape)
        new_x[:x.shape[0], :x.shape[1], :] = x
    else:
        print("Weight type Error:", w_type)
    if unsqueeze:
        new_x = new_x.squeeze()
    return new_x


# eg. math_comb(64, 2)
def comb_num(n, m):
    return math.factorial(n)//(math.factorial(n-m)*math.factorial(m))

def cal_none_overhead(raw_w_shape, sparsity):
    # raw_w = torch.flatten(raw_w, start_dim=1, end_dim=2).numpy()
    col_num = raw_w_shape[0]
    row_num = raw_w_shape[1] * raw_w_shape[2]
    weight_bit = 8
    return col_num * row_num * weight_bit

def cal_csr_overhead(raw_w_shape, sparsity):
    # raw_w = torch.flatten(raw_w, start_dim=1, end_dim=2).numpy()
    col_num = raw_w_shape[0]
    row_num = raw_w_shape[1] * raw_w_shape[2]
    weight_bit = 8

    scipy.random.seed(3)
    raw_w = scipy.sparse.random(raw_w_shape[0], raw_w_shape[1]*raw_w_shape[2], 
                        format='csr',density=1-sparsity, data_rvs=np.random.randn)
    # print(raw_w.indices)
    # print(raw_w.indptr)
    # print(raw_w.data)

    indics_overhead = len(raw_w.indices) * math.log(col_num, 2)
    indptr_overhead = len(raw_w.indptr) * math.log(col_num * row_num * (1-sparsity), 2)
    data_overhead   = len(raw_w.data) * weight_bit

    return indics_overhead + indptr_overhead #+ data_overhead

def cal_csc_overhead(raw_w_shape, sparsity):
    # raw_w = torch.flatten(raw_w, start_dim=1, end_dim=2).numpy()
    col_num = raw_w_shape[0]
    row_num = raw_w_shape[1] * raw_w_shape[2]
    weight_bit = 8

    scipy.random.seed(3)
    raw_w = scipy.sparse.random(raw_w_shape[0], raw_w_shape[1]*raw_w_shape[2], 
                        format='csc',density=1-sparsity, data_rvs=np.random.randn)
    # print(raw_w.indices)
    # print(raw_w.indptr)
    # print(raw_w.data)
    # scipy.sparse(raw_w)
    indics_overhead = len(raw_w.indices) * math.log(row_num, 2)
    indptr_overhead = len(raw_w.indptr) * math.log(col_num * row_num * (1-sparsity), 2)
    data_overhead   = len(raw_w.data) * weight_bit

    return indics_overhead + indptr_overhead #+ data_overhead

def cal_coo_overhead(raw_w_shape, sparsity, if_block=True):
    # raw_w = torch.flatten(raw_w, start_dim=1, end_dim=2).numpy()
    col_num = raw_w_shape[0]
    row_num = raw_w_shape[1] * raw_w_shape[2]
    weight_bit = 8

    scipy.random.seed(3)
    raw_w = scipy.sparse.random(raw_w_shape[0], raw_w_shape[1]*raw_w_shape[2], 
                        format='coo',density=1-sparsity, data_rvs=np.random.randn)
    # print(len(raw_w.col))
    # print(len(raw_w.row))
    # print(len(raw_w.data))
    # scipy.sparse(raw_w)
    if if_block == False:
        col_index_overhead = len(raw_w.col) * math.log(row_num, 2)
        row_index_overhead = len(raw_w.row) * math.log(col_num, 2)
    else:
        col_index_overhead = len(raw_w.col) * math.log(16, 2)
        row_index_overhead = len(raw_w.row) * math.log(16, 2)
    data_overhead  = len(raw_w.data) * weight_bit

    return col_index_overhead + row_index_overhead #+ data_overhead

def cal_rlc_overhead(raw_w_shape, sparsity, rlc_bit):
    col_num = raw_w_shape[0]
    row_num = raw_w_shape[1] * raw_w_shape[2]
    weight_bit = 8

    scipy.random.seed(3)
    raw_w = scipy.sparse.random(raw_w_shape[0], raw_w_shape[1]*raw_w_shape[2], 
                        format='csr',density=1-sparsity, data_rvs=np.random.randn).toarray()

    run_length = 2 ** rlc_bit
    run_overhead = 0
    weight_overhead = 0
    cnt = 0
    for col in range(col_num):
        for row in range(row_num):
            if raw_w[col][row] == 0:
                cnt += 1
                if cnt >= run_length:
                    cnt = 0
                    weight_overhead += weight_bit
                    run_overhead += rlc_bit
            else:
                cnt = 0
                weight_overhead += weight_bit
                run_overhead += rlc_bit

    return run_overhead #+ weight_overhead

def cal_bmp_overhead(raw_w_shape, sparsity):
    col_num = raw_w_shape[0]
    row_num = raw_w_shape[1] * raw_w_shape[2]
    weight_bit = 8

    bitmap_overhead = col_num * row_num
    weight_overhead = col_num * row_num * (1-sparsity) * weight_bit
    return bitmap_overhead #+ weight_overhead

def cal_ptn_overhead(raw_w_shape, sparsity, pattern_shape, pattern_num):
    col_num = raw_w_shape[0]
    row_num = raw_w_shape[1] * raw_w_shape[2]
    weight_bit = 8

    sub_matrix_num = (col_num * row_num) / (pattern_shape[0] * pattern_shape[1])
    pattern_coo_coding = pattern_num * (pattern_shape[0] * pattern_shape[1]) * (1-sparsity)

    pattern_coo_coding_overhead = pattern_coo_coding * (math.log(16, 2) + math.log(16, 2))
    pattern_idx_overhead = sub_matrix_num * math.log(pattern_num, 2)
    weight_overhead = col_num * row_num * (1-sparsity) * weight_bit
    return pattern_coo_coding_overhead + pattern_idx_overhead #+ weight_overhead


def coo_reserve_layer(raw_w, mask, coo_percent):
    p_w = raw_w * mask

    mask_r = 1 - mask
    coo_w = raw_w * mask_r

    w_num = raw_w.abs().flatten().size()[0]
    # print(w_num)
    # print(coo_percent)

    value, _ = torch.topk(coo_w.abs().flatten(), int(w_num*coo_percent))
    thre = abs(value[-1])
    zeros = torch.zeros_like(raw_w)
    coo_w = torch.where(abs(coo_w) < thre, zeros, raw_w)

    p_w = p_w + coo_w
    return p_w

def cal_overhead(net_arch, compression_rate):
    print('net_arch:',net_arch)
    print('compression_rate:',compression_rate)
    print('coding methos:coo, csr, csc, rlc2, rlc-4, bitmap, pattern')
    for r in compression_rate:
        sparsity = 1 - 1 / r
        cal_coo = 0
        cal_csr = 0
        cal_csc = 0
        cal_rlc2 = 0
        cal_rlc4 = 0
        cal_bit = 0
        cal_pat = 0
        for raw_w_shape,raw_w_num in net_arch:
            cal_coo += raw_w_num*cal_coo_overhead(raw_w_shape, sparsity, if_block=True)
            cal_csr += raw_w_num*cal_csr_overhead(raw_w_shape, sparsity)
            cal_csc += raw_w_num*cal_csc_overhead(raw_w_shape, sparsity)
            cal_rlc2 += raw_w_num*cal_rlc_overhead(raw_w_shape, sparsity, 2)
            cal_rlc4 += raw_w_num*cal_rlc_overhead(raw_w_shape, sparsity, 4)
            cal_bit += raw_w_num*cal_bmp_overhead(raw_w_shape, sparsity)
            cal_pat += raw_w_num*cal_ptn_overhead(raw_w_shape, sparsity, [8,8], 16)
        # print(cal_coo, cal_csr, cal_csc, cal_rlc, cal_rlc, cal_bit, cal_pat)
        print("compression_rate:", r,"coo:", cal_coo/8/1024,"rlc4:", cal_rlc4/8/1024, "pat:", cal_pat/8/1024)



    # import torch.nn as nn
    # from torch.utils.data import DataLoader
    # from dataset import VCTK
    # import dataset
    # from wavenet import WaveNet
    # from train import validate
    

    # vctk_val = VCTK(cfg, 'val')
    # val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    # # build model
    # model = WaveNet(num_classes=28, channels_in=20, dilations=[1,2,4,8,16])
    # model = nn.DataParallel(model)
    # model.cuda()
    # model_p = WaveNet(num_classes=28, channels_in=20, dilations=[1,2,4,8,16])
    # model_p = nn.DataParallel(model_p)
    # model_p.cuda()

    # model.load_state_dict(torch.load(
    #     '/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/fd_rtnl_16_8_8_24_l/debug/weights/dense_best.pth'), strict=True)
    # model_p.load_state_dict(torch.load(
    #     '/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/fd_rtnl_16_8_8_24_l/debug/weights/pruned_best.pth'), strict=True)
    
    # loss_fn = nn.CTCLoss(blank=27)
    # print(cal_sparsity(model))
    # print(cal_sparsity(model_p))
    # loss_val = validate(val_loader, model, loss_fn)
    # loss_val = validate(val_loader, model_p, loss_fn)

    # name_list = list()
    # para_list = list()
    # para_p_list = list()
    # for name, para in model.named_parameters():
    #     name_list.append(name)
    #     para_list.append(para)
    # for name, para in model_p.named_parameters():
    #     para_p_list.append(para)

    # a = model.state_dict()
    # for i, name in enumerate(name_list):
    #     if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
    #         raw_w = para_list[i]
    #         mask_w = para_p_list[i]
    #         cal_sparsity(model_p)
    #         ones = torch.ones_like(mask_w)
    #         zeros = torch.zeros_like(mask_w)
    #         mask = torch.where(abs(mask_w) <= 0, zeros, ones)
    #         p_w = coo_reserve_layer(raw_w, mask, 0.15)
    #         a[name] = p_w
    # model.load_state_dict(a)
    # loss_val = validate(val_loader, model, loss_fn)

    # a = model.state_dict()
    # for i, name in enumerate(name_list):
    #     if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
    #         raw_w = para_list[i]
    #         mask_w = para_p_list[i]
    #         cal_sparsity(model_p)
    #         ones = torch.ones_like(mask_w)
    #         zeros = torch.zeros_like(mask_w)
    #         mask = torch.where(abs(mask_w) <= 0, zeros, ones)
    #         p_w = coo_reserve_layer(raw_w, mask, 0.35)
    #         a[name] = p_w
    # model.load_state_dict(a)
    # loss_val = validate(val_loader, model, loss_fn)

    # a = model.state_dict()
    # for i, name in enumerate(name_list):
    #     if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
    #         raw_w = para_list[i]
    #         mask_w = para_p_list[i]
    #         cal_sparsity(model_p)
    #         ones = torch.ones_like(mask_w)
    #         zeros = torch.zeros_like(mask_w)
    #         mask = torch.where(abs(mask_w) <= 0, zeros, ones)
    #         p_w = coo_reserve_layer(raw_w, mask, 0.85)
    #         a[name] = p_w
    # model.load_state_dict(a)
    # loss_val = validate(val_loader, model, loss_fn)




if __name__ == "__main__":
    
    # print(loss_val)
    # raw_w_shape = (128,128,7)
    # raw_w_shape = (1632,36548,1)
    # raw_w_shape = (128,128,7)
    # compression_rate = [1, 2, 4, 8, 16, 32, 64]

    # for r in compression_rate:
    #     sparsity = 1 - 1 / r
    #     overhead = cal_rlc_overhead(raw_w_shape, sparsity, 8)
    #     print(r, overhead)

    # print("bitmap:", cal_bitmap_overhead(raw_w_shape, sparsity))
    # print("pattern:", cal_pattern_overhead(raw_w_shape, sparsity, [16,16], 16))
    # print("none:", cal_none_overhead(raw_w_shape, sparsity))
    # print("csr:", cal_csr_overhead(raw_w_shape, sparsity))
    # print("csc:", cal_csc_overhead(raw_w_shape, sparsity))
    # print("coo:", cal_coo_overhead(raw_w_shape, sparsity))
    # print("rcl4:", cal_rlc_overhead(raw_w_shape, sparsity, 4))
    # print("rcl2:", cal_rlc_overhead(raw_w_shape, sparsity, 2))
    # validate(val_loader, model, loss_fn)

    compression_rate = [1, 2, 4, 8, 16, 32, 64]
    lstm_compression_rate = [32.0, 28.4, 25.6, 23.3, 21.3, 19.7, 18.3, 17.1, 16.0]
    wavenet_compression_rate = [32.0, 28.4, 25.6, 23.3, 21.3, 19.7, 18.3, 17.1, 16.0]
    lstm_arch = [((512,512,1),12),((512,440,1),4)]
    wavenet_arch = [((128,128,7),30),((128,128,1),15)]
    # cal_overhead(wavenet_arch,compression_rate)
    # cal_overhead(lstm_arch,compression_rate)

    # cal_overhead(wavenet_arch, wavenet_compression_rate)
    # # # nnz:8-16  coo_nnz:0-8
    # for coo_nnz in np.arange(0, 2.25, 0.25):
    #     sparsity = 1 - coo_nnz/64
    #     cal_coo = 0
    #     for raw_w_shape,raw_w_num in wavenet_arch:
    #         cal_coo += raw_w_num*cal_coo_overhead(raw_w_shape, sparsity, if_block=True)
    #     print('coo_nnz:',coo_nnz,'coo_index:',cal_coo/8/1024)

    cal_overhead(lstm_arch, lstm_compression_rate)
    # nnz:2-4  coo_nnz:0-2
    for coo_nnz in np.arange(0, 2.25, 0.25):
        sparsity = 1 - coo_nnz/64
        cal_coo = 0
        for raw_w_shape,raw_w_num in lstm_arch:
            cal_coo += raw_w_num*cal_coo_overhead(raw_w_shape, sparsity, if_block=True)
        print('coo_nnz:',coo_nnz,'coo_index:',cal_coo/8/1024)

    # np.random.seed(0)
    # weights = []
    # for i in range(9):
    #     weights.append((np.random.rand(3,3)*10).round(decimals=1).flatten())

    # raw_w = np.random.rand(9,9)
    # for i in range(3):
    #     for j in range(3):
    #         raw_w[i*3:i*3+3,j*3:j*3+3] = weights[3*i+j].reshape(3,3)
    # raw_w = torch.from_numpy(raw_w).unsqueeze(2).cuda()


    # pattern_shape = [8, 8]
    # pattern_nnz = 8
    # stride = pattern_shape
    # pattern_num = 16
    # raw_w = torch.randn((512, 512, 1)).cuda()

    # pattern_set = find_top_k_by_kmeans(raw_w, pattern_num, pattern_shape, pattern_nnz, stride)
    # # print(pattern_set)
    # print(torch.abs(raw_w).sum())
    # mask = apply_patterns(raw_w, pattern_set)
    # print(mask.size(), raw_w.size())
    # prun_w = mask * raw_w
    # print(torch.abs(prun_w).sum())

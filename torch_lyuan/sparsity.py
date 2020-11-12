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
from sklearn.cluster import KMeans
import scipy.sparse

from itertools import combinations, permutations

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
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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
                if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                    if raw_w.size(0) == 128 and raw_w.size(1) == 128:
                        mask = apply_patterns(raw_w, cfg.fd_rtn_pattern_set[name])
                        p_w = raw_w * mask
                        a[name] = p_w
                    else:
                        a[name] = raw_w
                else:
                    a[name] = raw_w

        elif cfg.layer_or_model_wise == "m":
            for i, name in enumerate(name_list):
                raw_w = para_list[i]
                if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                    if raw_w.size(0) == 128 and raw_w.size(1) == 128:
                        mask = apply_patterns(raw_w, cfg.fd_rtn_pattern_set['all'])
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
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
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

def find_top_k_by_kmeans(raw_w, pattern_num, pattern_shape, pattern_nnz, stride):
    p_num_x = (raw_w.size(0) - pattern_shape[0])//stride[0] + 1
    p_num_y = (raw_w.size(1) - pattern_shape[1])//stride[1] + 1
    pattern_total_num = pattern_shape[0]*pattern_shape[1]
    pattern_set_len = comb_num(pattern_total_num, pattern_nnz)
    if pattern_set_len < pattern_num:
        pattern_num = pattern_set_len

    raw_w = torch.abs(raw_w)
    if raw_w.device.type == 'cuda':
        raw_w = raw_w.cpu()
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    # start_t = time.time()

    pattern_candidates = list()
    for k in range(raw_w.size(2)):
        for i in range(0, p_num_x):
            for j in range(0, p_num_y):
                sub_matrix = raw_w[i*stride[0]: i*stride[0] + pattern_shape[0],
                                          j*stride[1]: j*stride[1] + pattern_shape[1], k]
                value, _ = torch.topk(sub_matrix.abs().flatten(), pattern_nnz)
                zero_threshold = value[-1]

                ones = torch.ones_like(sub_matrix)
                zeros = torch.zeros_like(sub_matrix)
                pattern_candidate = torch.where(abs(sub_matrix) < zero_threshold, zeros, ones)
                pattern_candidates.append(pattern_candidate.numpy().flatten())

    clf = KMeans(n_clusters=pattern_num)
    clf.fit(pattern_candidates)  # 分组
    
    centers = clf.cluster_centers_ # 两组数据点的中心点

    pattern_set = list()
    for pattern in centers:
        pattern = torch.from_numpy(pattern)
        index = pattern.sort()[1][-pattern_nnz:]
        pattern = torch.zeros_like(pattern)
        for i in index:
            pattern[i] = 1

        pattern_set.append(pattern.reshape(pattern_shape[0], pattern_shape[1]))

    kernel = torch.zeros((len(pattern_set), 1, pattern_shape[0], pattern_shape[1])).cuda()
    for p_i, p in enumerate(pattern_set):
        kernel[p_i, 0, :, :] = pattern_set[p_i]

    return kernel



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

def apply_patterns(raw_w, kernel):
    # print(raw_w.size())
    raw_w = torch.abs(raw_w)
    # pattern_set = [(torch.from_numpy(p)) for p in pattern_set]
    start_t = time.time()
    # pattern_shape = [pattern_set[0].size(0), pattern_set[0].size(1)]
    pattern_shape = [kernel.size(2), kernel.size(3)]
    stride = (pattern_shape[0], pattern_shape[1])
    # print(pattern_shape)
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

    out_max = torch.max(out, dim=1)[0].unsqueeze(1).repeat(1,kernel.size(0),1,1)
    idx = torch.where(out >= out_max, torch.ones_like(out), torch.zeros_like(out))
    mask = torch.nn.functional.conv_transpose2d(idx, kernel, 
                bias=None, stride=stride, padding=0, output_padding=0, groups=1)
    mask = mask.squeeze(1)
    # print("apply one layer time==================", time.time() - start_t)
    if unsqueeze == True:
        mask = mask.squeeze(2)
    mask = mask.permute(1, 2, 0)
    # exit()
    return mask


# eg. math_comb(64, 2)
def comb_num(n, m):
    return math.factorial(n)//(math.factorial(n-m)*math.factorial(m))

def cal_csr_overhead(raw_w_shape, sparsity):
    # raw_w = torch.flatten(raw_w, start_dim=1, end_dim=2).numpy()

    scipy.random.seed(3)
    raw_w = scipy.sparse.random(raw_w_shape[0], raw_w_shape[1]*raw_w_shape[2], 
                        format='csr',density=1-sparsity, data_rvs=np.random.randn)
    # scipy.sparse(raw_w)


if __name__ == "__main__":

    # input = torch.tensor([[1,0], [0,1]]).unsqueeze(0).unsqueeze(0)
    # weight = torch.tensor([[[1,2], [3,4]]]).unsqueeze(0)
    # print(input.size())
    # print(weight.size())
    # output = torch.nn.functional.conv_transpose2d(input, weight, 
    #             bias=None, stride=2, padding=0, output_padding=0, groups=1)
    # print(output)
    # exit()

    np.random.seed(0)
    weights = []
    for i in range(9):
        weights.append((np.random.rand(3,3)*10).round(decimals=1).flatten())

    raw_w = np.random.rand(9,9)
    for i in range(3):
        for j in range(3):
            raw_w[i*3:i*3+3,j*3:j*3+3] = weights[3*i+j].reshape(3,3)
    raw_w = torch.from_numpy(raw_w).unsqueeze(2).cuda()

    pattern_shape = [3, 3]
    pattern_nnz = 3
    stride = pattern_shape
    pattern_num = 2
    # raw_w = torch.randn((9, 9, 1)).cuda()

    pattern_set = find_top_k_by_kmeans(raw_w, pattern_num, pattern_shape, pattern_nnz, stride)
    print(pattern_set)
    print(torch.abs(raw_w).sum())
    mask = apply_patterns(raw_w, pattern_set)
    # print(mask.size(), raw_w.size())
    prun_w = mask * raw_w
    print(torch.abs(prun_w).sum())


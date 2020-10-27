#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-18 15:31:19
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-27 10:43:42
# FilePath     : /speech-to-text-wavenet/torch_lyuan/sparsity.py
# Description  : 
#-------------------------------------------# 
import os
import numpy as np

import torch
import sys
import config_train as cfg



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
        pattern_shape  = cfg.coo_shape
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
                        assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                            assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                    oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] 
                                value, _ = torch.topk(part_w.abs().flatten(), coo_nnz)
                                thre = abs(value[-1])
                                zero = torch.zeros_like(part_w)
                                one = torch.ones_like(part_w)
                                part_mask = torch.where(abs(part_w) < thre, zero, one)
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
        pattern_shape  = cfg.pattern_shape
        pt_nnz  = cfg.pt_nnz
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
                        assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                            assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                                not_part_w = not_p_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                    oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] 
                                value, _ = torch.topk(not_part_w.abs().flatten(), coo_nnz)
                                thre = abs(value[-1])
                                zero = torch.zeros_like(not_part_w)
                                one = torch.ones_like(not_part_w)
                                part_mask = torch.where(abs(not_part_w) < thre, zero, one)
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
                    assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                    for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                        assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
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
                    assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                    for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                        assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
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
        raw_w, patterns_layer = find_pattern_certain_nnz_layer(raw_w, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun)
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
        for ic_p in range(raw_w.size(0)// pattern_shape[0]):
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
        sorted_patterns = sorted(patterns.keys(),key=lambda item:patterns[item],reverse=True)
        selected_pattern_list = sorted_patterns[:pattern_num]
        raw_w = pattern_prun_certain_nnz_layer(raw_w, selected_pattern_list, pattern_shape)
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
    selected_pattern_list = [torch.from_numpy(np.fromstring(selected_pattern_list[i],dtype=np.float32)).cuda().reshape(pattern_shape) for i in range(len(selected_pattern_list))]
    for k in range(raw_w.size(2)):
        for ic_p in range(raw_w.size(0)// pattern_shape[0]):
            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                            oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k].abs()
                pattern_index = max(range(len(selected_pattern_list)),key=lambda i:torch.sum(selected_pattern_list[i]*(part_w)))
                raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                            oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = selected_pattern_list[pattern_index]
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
            for ic_p in range(int(raw_w.size(0)/ pattern_shape[0])):
                for oc_p in range(int(raw_w.size(1) / pattern_shape[1])):
                    part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                        oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k]
                    zero = torch.zeros_like(part_w)
                    one = torch.ones_like(part_w)
                    pattern = torch.where(part_w == 0, zero, one).cpu().numpy().tostring()
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
    for k,v in x.items():
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
            zero_cnt += w.flatten().size()[0] - torch.nonzero(w).size()[0]
            all_cnt += w.flatten().size()[0]

    return zero_cnt/all_cnt


#----------------description----------------# 
# description: find pattern by similarity
#                step 1: generate pattern candidates
#                step 2: calculate the output score mask (remove patterns)
#                step 3: return top-k patterns
# param {*} raw_w
# param {*} pattern_num
# param {*} pattern_shape
# param {*} zero_threshold
# param {*} score_threshold
# return {*}
#-------------------------------------------# 
def find_pattern_by_similarity(raw_w, pattern_num, pattern_shape, zero_threshold, score_threshold):
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)

    # mask = torch.zeros_like(raw_w).cuda()

    one = torch.ones_like(raw_w)
    zero = torch.zeros_like(raw_w)
    mask = torch.where(abs(raw_w) <= zero_threshold, zero, one).cuda()

    # get pattern candidates
    pattern_candidates = list()
    idx_to_ijk = dict()
    idx = 0
    for k in range(raw_w.size(2)):
        for i in range(raw_w.size(0) - pattern_shape[0] +1):
            for j in range(raw_w.size(1) - pattern_shape[1] +1):
                idx_to_ijk[idx] = [i, j, k]
                # part_w = raw_w[i: i + pattern_shape[0],
                #                 j: j + pattern_shape[1], k]
                # one = torch.ones_like(part_w)
                # zero = torch.zeros_like(part_w)
                # pattern_candidate = torch.where(abs(part_w) <= zero_threshold, zero, one).cuda()
                pattern_candidate = mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k]
                # mask[i: i + pattern_shape[0],
                #     j: j + pattern_shape[1], k] = pattern_candidate

                pattern_candidates.append(pattern_candidate)
                idx += 1


    # output score maps
    score_maps = list()
    pattern_match_num_dict = dict()
    print(len(pattern_candidates))
    pattern_candidates, pattern_sort_index = sort_pattern_candidates(pattern_candidates)
    remove_bitmap = torch.zeros((raw_w.size(0) - pattern_shape[0] +1, raw_w.size(1) - pattern_shape[1] +1, raw_w.size(2)))
    
    print("sorted: ", len(pattern_candidates))
    for p_idx, p in enumerate(pattern_candidates):
        p_idx = pattern_sort_index[p_idx]

        p_i = idx_to_ijk[p_idx][0]
        p_j = idx_to_ijk[p_idx][1]
        p_k = idx_to_ijk[p_idx][2]

        # print(p_i, p_j, p_k)
        if remove_bitmap[p_i, p_j, p_k] == 0:
            score_map = torch.zeros((raw_w.size(0) - pattern_shape[0] +1, raw_w.size(1) - pattern_shape[1] +1, raw_w.size(2)))

            for k in range(raw_w.size(2)):
                for i in range(raw_w.size(0) - pattern_shape[0] +1):
                    for j in range(raw_w.size(1) - pattern_shape[1] +1):
                        if remove_bitmap[i, j, k] == 1:
                            score_map[i, j, k] = 0
                        else:
                            score_map[i, j, k] = (p * mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k]).sum()

                            # if i == p_i and j == p_j and k == p_k:
                            #     # print(pattern_candidates[p_idx])
                            #     print(p)
                            #     print(mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k])
                            #     print((p * mask[i: i + pattern_shape[0], j: j + pattern_shape[1], k]).sum())
                            
            # score_maps.append(score_map)
            score_max = score_map.max()
            assert score_max <= p.sum(), f"{score_max} {p} {p.sum()}"

            # remove the candidate score match the score threshold
            zeros = torch.zeros_like(remove_bitmap)
            ones = torch.ones_like(remove_bitmap)
            # print(remove_bitmap)
            # print(score_max, score_threshold)
            remove_bitmap_add = torch.where(score_map >= abs(score_max-score_threshold), ones, zeros)
            # print(torch.nonzero(remove_bitmap_add))
            remove_bitmap = torch.where(remove_bitmap_add >= 1, ones, remove_bitmap)

            # print(remove_bitmap)
            match_num = remove_bitmap_add.sum()
            print(idx_to_ijk[p_idx], ",current_pattern_nnz:", int(p.sum()), 
                                    ",output_max:", int(score_max), 
                                    ",score:", int(match_num), 
                                    ",removed:", int(remove_bitmap.sum()))
            pattern_match_num_dict[p.cpu().numpy().tostring()] = match_num
        else:
            pass

    print(len(pattern_match_num_dict))
    # collect top-pattern_num patterns
    if len(pattern_match_num_dict.items()) < pattern_num:
        pattern_num = len(pattern_match_num_dict)
    patterns = list()
    pattern_match_num_dict_sorted = sorted(pattern_match_num_dict.items(), key = lambda k: k[pattern_num])
    for p, score in pattern_match_num_dict_sorted:
        p = np.frombuffer(p, dtype=np.float32).reshape(pattern_shape)
        print(p, score)
        patterns.append(p)

    return patterns
    

def sort_pattern_candidates(pattern_candidates):
    pattern_candidates_sorted = pattern_candidates.copy()
    pattern_sort_index = sorted(range(len(pattern_candidates)), key=lambda k: pattern_candidates[k].sum(), reverse=True)
    # print(pattern_sort_index)
    pattern_candidates_sorted = [pattern_candidates_sorted[i] for i in pattern_sort_index]
    # for i in range(len(pattern_candidates)-1):
    #     for j in range(len(pattern_candidates)-1-i):
    #         if pattern_candidates[j].sum() > pattern_candidates[j+1].sum():
    #             pattern_candidates[j], pattern_candidates[j+1] = pattern_candidates[j+1], pattern_candidates[j]

    return pattern_candidates_sorted, pattern_sort_index
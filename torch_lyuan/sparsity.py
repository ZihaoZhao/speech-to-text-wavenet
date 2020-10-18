import os
import numpy as np

import torch
import config_train as cfg

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
            mask = cfg.pattern_mask[name]
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
            mask = cfg.pattern_mask[name]
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


    return patterns_mask

def save_pattern():
    
    pattern_num = 16
    pattern_shape = [16, 16]
    pattern_nnz = 32
    sparsity = pattern_nnz / (pattern_shape[0] * pattern_shape[1])
    patterns = dict()

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
        for k in raw_w.size(0):
            for ic_p in raw_w.size(1)/ pattern_shape[0]:
                for oc_p in raw_w.size(2) / pattern_shape[1]:
                    part_w = raw_w[k, ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                        oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1]]
                    value, _ = torch.topk(part_w.abs().flatten(), pattern_nnz)

                    # pruning
                    thre = abs(value[-1])
                    zero = torch.zeros_like(part_w)
                    part_w_p = torch.where(abs(part_w) < thre, zero, part_w)
                    raw_w[k, ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                        oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1]] = part_w_p

                    # save the pattern
                    ones = torch.ones_like(part_w)
                    pattern = torch.where(abs(part_w) < thre, zero, ones)
                    if pattern not in patterns.keys():
                        patterns[pattern] = 1
                    else:
                        patterns[pattern] += 1

    return patterns
    
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
        if name.split(".")[-2] != "bn":
            zero_cnt += w.flatten().size()[0] - torch.nonzero(w).size()[0]
            all_cnt += w.flatten().size()[0]

    return zero_cnt/all_cnt

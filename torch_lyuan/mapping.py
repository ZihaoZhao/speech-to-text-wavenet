#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-12-20 11:52:22
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-12-21 22:24:31
# FilePath     : /speech-to-text-wavenet/torch_lyuan/mapping.py
# Description  : 
#-------------------------------------------# 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as  rnn_utils
import deepdish as dd

import config_train as cfg
from dataset import VCTK
import dataset
from wavenet import WaveNet
from sparsity import *
import utils
import visualize as vis

from ctcdecode import CTCBeamDecoder

from tensorboardX import SummaryWriter
import os
import numpy as np

import time
import argparse
from write_excel import *

model_pth = "/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/fd_rtnl_16_8_8_4_0_l_bn_ok/debug/weights/best.pth"
pattern_dir = "/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/fd_rtnl_16_8_8_4_0_l_bn_ok/debug/patterns"
save_dir = "/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/fd_rtnl_16_8_8_4_0_l_bn_ok/debug/weights/weight_txt"
# model_pth = "/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/fd_rtnl_16_8_8_4_0_l/debug/weights/best.pth"

def main():
    print("Mapping...")
    # build model
    model = WaveNet(num_classes=28, channels_in=40, dilations=[1,2,4,8,16])
    model = nn.DataParallel(model)
    model.cuda()

    model.load_state_dict(torch.load(model_pth), strict=True)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_weight_txt(model, save_dir)

def save_weight_txt(model, folder):

    name_list = list()
    para_list = list()

    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    for i, name in enumerate(name_list):
        # pytorch OC, IC, K
        # C model K, IC, OC        

        raw_w = para_list[i]
        raw_w_save = np.array(raw_w.cpu().detach())
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
            raw_w_save = raw_w_save.transpose((2, 1, 0))
        np.savetxt(os.path.join(folder, name + '.txt'), raw_w_save.flatten())
        
def read_txt():
    layer = "/module.resnet_block_0.0.conv_filter.dilation_conv1d.weight.txt"
    # layer = "/module.resnet_block_2.4.conv_filter.dilation_conv1d.weight.txt"
    pattern_txt = pattern_dir + layer
    weight_txt  = save_dir + layer
    pattern = np.loadtxt(pattern_txt).reshape((16, 8, 8))
    weight = np.loadtxt(weight_txt).reshape((7, 128, 128))
    # weight = np.loadtxt(weight_txt).reshape((1, 40, 128))
    for i in range(0, 16):
        for j in range(0, 8):
            for k in range(0, 8):
                print(pattern[i, j, k], end = ' ')
            print(" ")
        print(i)
    print("w")
    for j in range(0, 8):
        for k in range(0, 8):
            print(weight[0, j, k], end = ' ')
        print(" ")

if __name__ == "__main__":
    main()
    read_txt()
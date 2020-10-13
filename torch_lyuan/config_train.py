#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-13 10:05:00
# FilePath     : /speech-to-text-wavenet/torch_lyuan/config_train.py
# Description  : 
#-------------------------------------------# 

distributed = True # distributed training, must be True
mode = None # set at dataloader, must be None

# workdir = '/lyuan/code/speech-to-text-wavenet/torch_lyuan/exp/debug' # all output you need will be saved here, include ckpt.pth.

# # path setting
# dataset = '/lyuan/dataset/VCTK'
# datalist = '/lyuan/code/speech-to-text-wavenet/data/list.json'


workdir = '/zhzhao/code/wavenet_torch/torch_lyuan/exp_bs32/debug' # all output you need will be saved here, include ckpt.pth.

# path setting
dataset = '/zhzhao/VCTK'
datalist = '/zhzhao/code/wavenet_torch/data/list.json'

batch_size = 32 # reconmendate 32
load_from = ''
epochs = 1000
lr = 1e-5

#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : ,: Zihao Zhao
# LastEditTime : ,: 2020-10-21 15:00:28
# FilePath     : ,: /speech-to-text-wavenet/torch_lyuan/config_train.py
# Description  : 
#-------------------------------------------# 
user = 'zzh'

distributed = True # distributed training, must be True
mode = None # set at dataloader, must be None
resume = True

# workdir = '/lyuan/code/speech-to-text-wavenet/torch_lyuan/exp/debug' # all output you need will be saved here, include ckpt.pth.

# # path setting
# dataset = '/lyuan/dataset/VCTK'
# datalist = '/lyuan/code/speech-to-text-wavenet/data/list.json'

exp_name = 'dense_32'
# workdir = '/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/' + exp_name + '/debug' # all output you need will be saved here, include ckpt.pth.

vis_dir = '/zhzhao/code/wavenet_torch/torch_lyuan/vis'

# path setting
dataset = '/zhzhao/VCTK'
datalist = '/zhzhao/code/wavenet_torch/data/list.json'

# sparse_mode = 'thre_pruning'#'dense'#'thre_pruning'
# pruning_thre = 0.05   ### zero:971, all:2560

sparse_mode = 'dense'#'dense'#'thre_pruning'
sparsity = 0.2   ### zero:971, all:2560

batch_size = 32 # reconmendate 32
load_from = ''
epochs = 1000
lr = 1e-2

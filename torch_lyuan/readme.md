using distributed to speed up training
# how to run
1. setting dataset path and other config in config_train.py
2. fix node numbers and the path of train_distributed.py in dist_train.sh
3. bash dist_train.sh


# ctc decoder
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .

# run exp

ps -ef|grep python|cut -c 9-15 |xargs kill -s9

CUDA_VISIBLE_DEVICES=0 screen python train.py --exp dense_32

CUDA_VISIBLE_DEVICES=1 screen python train.py --exp sparse_020_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.2

CUDA_VISIBLE_DEVICES=0 screen python train.py --exp sparse_050_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.5

CUDA_VISIBLE_DEVICES=1 screen python train.py --exp sparse_080_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.8


#tmp

#----------------description----------------#
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-14 15:00:22
# FilePath     : /speech-to-text-wavenet/torch_lyuan/train.py
# Description  : 
#-------------------------------------------# 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import config_train as cfg
from dataset import VCTK
from wavenet import WaveNet
import utils

from ctcdecode import CTCBeamDecoder

from tensorboardX import SummaryWriter
import os
import numpy as np

import argparse


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='SNN for BMI.')
    parser.add_argument('--resume', action='store_true', help='resume from exp_name/last.pth', default=True)
    parser.add_argument('--exp', type=str, help='exp dir', default="default")
    parser.add_argument('--sparse_mode', type=str, help='dense, sparse_pruning, thre_pruning', default="dense")
    parser.add_argument('--sparsity', type=float, help='0.2, 0.4, 0.8', default=0.2)
    parser.add_argument('--load_from', type=str, help='.pth', default="")

    args = parser.parse_args()
    return args

def train(train_loader, scheduler, model, loss_fn, val_loader, writer=None):
    
    vocabulary = utils.Data.vocabulary
    decoder = CTCBeamDecoder(
        vocabulary,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=False
    )

    weights_dir = os.path.join(cfg.workdir, 'weights')
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    model.train()

    if cfg.resume and os.path.exists(cfg.workdir + '/weights/last.pth'):
        model.load_state_dict(torch.load(cfg.workdir + '/weights/last.pth'))
        print("loading", cfg.workdir + '/weights/last.pth')

    if os.path.exists(cfg.load_from):
        model.load_state_dict(torch.load(cfg.load_from))
        print("loading", cfg.load_from)
        

    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'Training epoch {epoch}')
        _loss = 0.0
        step_cnt = 0
        
        model = pruning(model, cfg.sparse_mode)
        sparsity = cal_sparsity(model)
        print("sparsity:", sparsity)
        for data in train_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            logits = model(wave)
            logits = logits.permute(2, 0, 1) # [520, 32, 28]
            logits = F.log_softmax(logits, dim=2)
            text = data['text'].cuda()
            loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data   

            ### TODO evluate here
            logits = logits.permute(1, 0, 2)

            if step_cnt % int(12000/cfg.batch_size) == 1:
                print("Epoch", epoch,
                        ", train step", step_cnt, "/", len(train_loader),
                        ", loss: ", round(float(_loss.data/step_cnt), 5))
                torch.save(model.state_dict(), cfg.workdir+'/weights/last.pth')
                beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
                voc = np.tile(vocabulary, (cfg.batch_size, 1))
                pred = np.take(voc, beam_results[:,0,:].data.numpy())
                # print('text: ', text.shape)
                text_np = np.take(voc, text.data.cpu().numpy().astype(int))
                tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))

            step_cnt += 1
            
        _loss /= len(train_loader)
        writer.add_scalar('train/loss', _loss, epoch)
        torch.cuda.empty_cache()

        loss_val = validate(val_loader, model, loss_fn)
        writer.add_scalar('val/loss', loss_val, epoch)

        if loss_val < best_loss:
            torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
            print("saved", cfg.workdir+'/weights/best.pth')
            best_loss = loss_val


def pruning(model, sparse_mode="dense"):
    if sparse_mode == 'thre_pruning':
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
            if name.split(".")[-2] != "bn":
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
            value, _ = torch.topk(raw_w.abs().flatten(), w_num - zero_num)
            thre = abs(value[-1])
            zero = torch.zeros_like(raw_w)
            p_w = torch.where(abs(raw_w) < thre, zero, raw_w)
            
            # if name.split(".")[-2] != "bn":
            #     print(name, float(thre))
            #     print("all: ", raw_w.flatten().size()[0])
            #     print("zero: ", raw_w.flatten().size()[0] - torch.nonzero(p_w).size(0))
            #     print(" ")

            zero_cnt += torch.nonzero(p_w).size()[0]
            all_cnt += torch.nonzero(raw_w).size()[0]
            a[name] = p_w
            
        model.load_state_dict(a)
        
    return model

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


def validate(val_loader, model, loss_fn):
    model.eval()
    _loss = 0.0
    step_cnt = 0
    for data in val_loader:
        wave = data['wave'].cuda()  # [1, 128, 109]
        logits = model(wave)
        logits = logits.permute(2, 0, 1)
        logits = F.log_softmax(logits, dim=2)
        text = data['text'].cuda()
        loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
        _loss += loss.data
        # print(loss)
        step_cnt += 1
        # if cnt % 10 == 0:
    print("Val step", step_cnt, "/", len(val_loader),
            ", loss: ", round(float(_loss.data/step_cnt), 5))

    
    return _loss/len(val_loader)


def main():
    args = parse_args()
    cfg.resume      = args.resume
    cfg.exp_name    = args.exp
    cfg.workdir     = '/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/' + args.exp + '/debug'
    cfg.sparse_mode = args.sparse_mode
    cfg.sparsity    = args.sparsity
    cfg.load_from   = args.load_from
    
    print('initial training...')
    print(f'work_dir:{cfg.workdir}, \n\
            pretrained: {cfg.load_from},  \n\
            batch_size: {cfg.batch_size}, \n\
            lr        : {cfg.lr},         \n\
            epochs    : {cfg.epochs},     \n\
            sparse    : {cfg.sparse_mode}')
    writer = SummaryWriter(log_dir=cfg.workdir+'/runs')

    # build train data
    vctk_train = VCTK(cfg, 'train')
    train_loader = DataLoader(vctk_train,batch_size=cfg.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    vctk_val = VCTK(cfg, 'val')
    val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    # build model
    model = WaveNet(num_classes=28, channels_in=20, dilations=[1,2,4,8,16])
    model = nn.DataParallel(model)
    model.cuda()

    # build loss
    loss_fn = nn.CTCLoss()

    #
    scheduler = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(train_step, milestones=[50, 150, 250], gamma=0.5)

    # train
    train(train_loader, scheduler, model, loss_fn, val_loader, writer)
    # val
    # loss = validate(val_loader, scheduler, model, loss_fn)

if __name__ == '__main__':
    main()

#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-14 10:56:26
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

from tensorboardX import SummaryWriter
import os


def train(train_loader, scheduler, model, loss_fn, val_loader, writer=None):
    weights_dir = os.path.join(cfg.workdir, 'weights')
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    model.train()

    if os.path.exists('/zhzhao/code/wavenet_torch/torch_lyuan/exp_thre_pruning_comp_32/debug/weights/last.pth'):
        model.load_state_dict(torch.load('/zhzhao/code/wavenet_torch/torch_lyuan/exp_thre_pruning_comp_32/debug/weights/last.pth'))
        print("loading", '/zhzhao/code/wavenet_torch/torch_lyuan/exp_thre_pruning_comp_32/debug/weights/last.pth')
    # if os.path.exists(cfg.workdir + '/weights/last.pth'):
    #     model.load_state_dict(torch.load(cfg.workdir + '/weights/last.pth'))
    #     print("loading", cfg.workdir + '/weights/last.pth')
        
    model = pruning(model, cfg.sparse_mode)
    sparsity = cal_sparsity(model)
    print("sparsity:", sparsity)

    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'training epoch{epoch}')
        _loss = 0.0
        step_cnt = 0
        
        for data in train_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            logits = model(wave)
            logits = logits.permute(2, 0, 1)
            logits = F.log_softmax(logits, dim=2)
            text = data['text'].cuda()
            loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data
            if not _loss.data/(step_cnt+1) <100:
                print(data['name'])
                print(data)
                print(logits)
                print(text)
                exit()

            if step_cnt % int(10000/cfg.batch_size) == 0:
                print("Epoch", epoch,
                        ", train step", step_cnt, "/", len(train_loader),
                        ", loss: ", round(float(_loss.data/step_cnt), 5))
                torch.save(model.state_dict(), cfg.workdir+'/weights/last.pth')

            step_cnt += 1
            
        _loss /= len(train_loader)

        loss_val = validate(val_loader, model, loss_fn)

        writer.add_scalar('train/loss', _loss, epoch)
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

        a = torch.load(cfg.workdir+'/weights/last.pth')
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

        a = torch.load(cfg.workdir+'/weights/last.pth')
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

    #TODO evluate
    
    return _loss/len(val_loader)


def main():
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

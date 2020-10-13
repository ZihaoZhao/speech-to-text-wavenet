#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-13 11:26:51
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

    if os.path.exists(cfg.workdir + '/weights/last.pth'):
        model.load_state_dict(torch.load(cfg.workdir + '/weights/last.pth'))
        print("loading", cfg.workdir + '/weights/last.pth')
        
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'training epoch{epoch}')
        _loss = 0.0
        cnt = 0
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
            cnt += 1
            if not _loss.data/cnt <100:
                print(data['name'])
                print(data)
                print(logits)
                print(text)
                exit()
            if cnt % 1000 == 0:
                print("Epoch", epoch,
                        ", train step", cnt, "/", len(train_loader),
                        ", loss: ", round(float(_loss.data/cnt), 5))
                torch.save(model.state_dict(), cfg.workdir+'/weights/last.pth')
        _loss /= len(train_loader)

        loss_val = validate(val_loader, model, loss_fn)

        writer.add_scalar('train/loss', _loss, epoch)
        writer.add_scalar('val/loss', loss_val, epoch)

        if loss_val < best_loss:
            torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
            print("saved", cfg.workdir+'/weights/best.pth')
            best_loss = loss_val


def validate(val_loader, model, loss_fn):
    model.eval()
    _loss = 0.0
    cnt = 0
    for data in val_loader:
        wave = data['wave'].cuda()  # [1, 128, 109]
        logits = model(wave)
        logits = logits.permute(2, 0, 1)
        logits = F.log_softmax(logits, dim=2)
        text = data['text'].cuda()
        loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
        _loss += loss.data
        print(loss)
        cnt += 1
        # if cnt % 10 == 0:
    print("Val step", cnt, "/", len(val_loader),
            ", loss: ", round(float(_loss.data/cnt), 5))

    #TODO evluate
    
    return _loss/len(val_loader)


def main():
    print('initial training...')
    print(f'work_dir:{cfg.workdir}, pretrained:{cfg.load_from}, batch_size:{cfg.batch_size} lr:{cfg.lr}, epochs:{cfg.epochs}, sparse:{cfg.sparse}')
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

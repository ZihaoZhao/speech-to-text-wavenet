import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

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
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'training epoch{epoch}')
        _loss = 0.0
        cnt = 0
        for data in train_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            logits = model(wave)
            logits = logits.permute(2, 0, 1)
            text = data['text'].cuda()
            loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data
            cnt += 1
            if cnt % 1000 == 0:
                print("Epoch", epoch,
                        ", train step", cnt, "/", len(train_loader),
                        ", loss: ", round(float(_loss.data/cnt), 5))
        _loss /= len(train_loader)

        loss_val = validate(val_loader, model, loss_fn)

        writer.add_scalar('train/loss', _loss, epoch)
        writer.add_scalar('val/loss', loss_val, epoch)

        if loss_val < best_loss:
            torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
            best_loss = loss_val


def validate(val_loader, model, loss_fn):
    model.eval()
    _loss = 0.0
    cnt = 0
    for data in val_loader:
        wave = data['wave'].cuda()  # [1, 128, 109]
        logits = model(wave)
        logits = logits.permute(2, 0, 1)
        text = data['text'].cuda()
        loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
        scheduler.zero_grad()
        loss.backward()
        scheduler.step()
        _loss += loss.data
        cnt += 1
        if cnt % 500 == 0:
            print("Val step", cnt, "/", len(val_loader),
                    ", loss: ", round(float(_loss.data/cnt), 5))
    return _loss/len(val_loader)


def main():
    print('initial training...')
    print(f'work_dir:{cfg.workdir}, pretrained:{cfg.load_from}, batch_size:{cfg.batch_size}, lr:{cfg.lr}, epochs:{cfg.epochs}')
    writer = SummaryWriter(log_dir=cfg.workdir+'/runs')

    # build train data
    vctk_train = VCTK(cfg, 'train')
    train_loader = DataLoader(vctk_train,batch_size=cfg.batch_size, num_workers=32, shuffle=True,)

    vctk_val = VCTK(cfg, 'val')
    val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, num_workers=32, shuffle=False,)

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

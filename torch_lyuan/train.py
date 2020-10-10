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
        print('begin...')
        _loss = 0.0
        for data in train_loader:
            wave = data['wave'].cuda() # [1, 128, 109]
            logits = model(wave)
            logits = logits.permute(2,0,1)
            [_, N, _] = logits.shape
            text = data['text'].cuda()
            loss = loss_fn(logits, text, torch.tensor(N), torch.tensor(N))
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data
        _loss /= len(train_loader)
        print('finish a epoch')

        loss_val = validate(val_loader, model, loss_fn)

        writer.add_scalar('train/loss', _loss, epoch)
        writer.add_scalar('val/loss', loss_val, epoch)

        if loss_val < best_loss:
            torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
            best_loss = loss_val


def validate(val_loader, model, loss_fn):
    model.eval()
    loss_ = []
    for data in val_loader:
        data = data.cuda()
        x = data[:, :-1]

        logits = model(x)
        y = data[:, -logits.size(2):]
        loss = loss_fn(logits.transpose(1, 2).contiguous().view(-1, 256), y.view(-1))
        loss_.append(loss.data[0])
    return sum(loss_)


def main():
    writer = SummaryWriter(log_dir=cfg.workdir+'/runs')

    # build train data
    vctk_train = VCTK(cfg, 'train')
    train_loader = DataLoader(vctk_train,batch_size=1, shuffle=True,)

    vctk_val = VCTK(cfg, 'val')
    val_loader = DataLoader(vctk_val, batch_size=1, shuffle=False,)

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
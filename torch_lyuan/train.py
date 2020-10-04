import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torch.optim as optim

import config_train as CONFIG
from dataset import VCTK
from utils import collate_fn_
from wavenet import WaveNet

from tensorboardX import SummaryWriter


def train(train_loader, scheduler, model, loss_fn, val_loader, writer=None):
    weights_dir = os.path.join(CONFIG.workdir, 'weights')
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    model.train()
    best_loss = float('inf')
    for epoch in range(CONFIG.epoch):
        _loss = 0.0
        scheduler.step()
        for data in train_loader:
            data = data.cuda()
            input = data['input']
            logits = model(input)
            target = data['target']
            loss = loss_fn(logits, target, len(logits[-1]), len(target))
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data
        _loss /= len(train_loader)

        loss_val = validate(val_loader, model, loss_fn)

        writer.add_scalar('train/loss', _loss, epoch)
        writer.add_scalar('val/loss', loss_val, epoch)

        if loss_val < best_loss:
            torch.save(model.state_dict(), CONFIG.workdir+'/weights/best.pth')
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
    writer = SummaryWriter(log_dir=CONFIG.workdir)

    # build train data
    vctk_train = VCTK(path=CONFIG.dataset)
    train_loader = DataLoader(vctk_train,batch_size=1, shuffle=True,collate_fn=collate_fn_)

    vctk_val = VCTK(path=CONFIG.dataset)
    val_loader = DataLoader(vctk_val, batch_size=1, shuffle=True, collate_fn=collate_fn_)

    # build model
    model = WaveNet(num_classes=8, channels_in=16, dilations=[1,2,4]).cuda()
    model = nn.DataParallel(model)

    # build loss
    loss_fn = nn.CTCLoss()

    #
    scheduler = optim.Adam(model.parameters(), lr=CONFIG.lr, eps=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(train_step, milestones=[50, 150, 250], gamma=0.5)

    # train
    train(train_loader, scheduler, model, loss_fn, val_loader, writer)
    # val
    # loss = validate(val_loader, scheduler, model, loss_fn)

if __name__ == '__main__':
    main()
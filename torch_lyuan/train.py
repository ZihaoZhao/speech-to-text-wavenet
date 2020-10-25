#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-23 17:04:16
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
import dataset
from wavenet import WaveNet
from sparsity import *
import utils
import visualize as vis

from ctcdecode import CTCBeamDecoder

from tensorboardX import SummaryWriter
import os
import numpy as np

import argparse
from write_excel import write_excel

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='WaveNet for speech recognition.')
    parser.add_argument('--resume', action='store_true', help='resume from exp_name/best.pth', default=False)
    parser.add_argument('--vis_mask', action='store_true', help='visualize and save masks', default=False)
    parser.add_argument('--vis_pattern', action='store_true', help='visualize and save patterns', default=False)
    parser.add_argument('--exp', type=str, help='exp dir', default="default")
    parser.add_argument('--sparse_mode', type=str, help='dense, sparse_pruning, thre_pruning, pattern_pruning', default="dense")
    parser.add_argument('--sparsity', type=float, help='0.2, 0.4, 0.8', default=0.2)
    parser.add_argument('--pattern_para', type=str, help='[pt_num_pt_shape0_pt_shape1_nnz]', default='16_16_16_128')
    parser.add_argument('--coo_para', type=str, help='[pt_shape0, pt_shape1, nnz]', default='8_8_32')
    parser.add_argument('--ptcoo_para', type=str, help='[pt_num, pt_shape0, pt_shape1, pt_nnz, coo_nnz]', default='16_16_16_128_64')
    parser.add_argument('--batch_size', type=int, help='1, 16, 32', default=32)
    parser.add_argument('--lr', type=float, help='0.001 for tensorflow', default=0.001)
    parser.add_argument('--load_from', type=str, help='.pth', default="/z")
    parser.add_argument('--skip_exist', action='store_true', help='if exist', default=False)
    parser.add_argument('--save_excel', type=str, help='exp.xls', default="default.xls")
    parser.add_argument('--find_pattern', action='store_true', help='find_pattern', default=False)

    args = parser.parse_args()
    return args

def train(train_loader, scheduler, model, loss_fn, val_loader, writer=None):
    
    decoder_vocabulary = utils.Data.decoder_vocabulary
    vocabulary = utils.Data.vocabulary
    decoder = CTCBeamDecoder(
        decoder_vocabulary,
        #"_abcdefghijklmopqrstuvwxyz_",
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=27,
        log_probs_input=True
    )

        
    train_loss_list = list()
    val_loss_list = list()

    # prefetcher = data_prefetcher(train_loader)
    # data = prefetcher.next()
    
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'Training epoch {epoch}')
        _loss = 0.0
        step_cnt = 0
        
        # sparsity = cal_sparsity(model)
        # print("sparsity:", sparsity)
        for data in train_loader:
            # data = prefetcher.next()
            wave = data['wave'].cuda()  # [1, 128, 109]
            model = pruning(model, cfg.sparse_mode)

            if epoch == 0 and step_cnt == 0:
                loss_val = validate(val_loader, model, loss_fn)
                writer.add_scalar('val/loss', loss_val, epoch)
                val_loss_list.append(float(loss_val))
                
            logits = model(wave)
            logits = logits.permute(2, 0, 1)
            logits = F.log_softmax(logits, dim=2)
            # print(logits[:, 0, :].max(1))
            # for l in logits[:, 0, :].max(1)[1]:
            #     print(vocabulary[l], end='')
            # print(data['text'][0])
            # logits = F.softmax(logits, dim=2)
            if data['text'].size(0) == cfg.batch_size:
                for i in range(cfg.batch_size):
                    if i == 0:
                        text = data['text'][i][0:data['length_text'][i]].cuda()
                        # print(data['text'].size())
                        # print(data['length_text'][i])
                    else:
                        text = torch.cat([text, 
                                    data['text'][i][0: data['length_text'][i]].cuda()])
            else:
                continue


            try:
                loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
                scheduler.zero_grad()
                loss.backward()
                scheduler.step()
                # print(data['length_text'])
                # print(data['length_text'].size().data)
                _loss += loss.data


                if epoch == 0 and step_cnt == 10:
                    writer.add_scalar('train/loss', _loss, epoch)
                    train_loss_list.append(float(_loss))

                if step_cnt % int(12000/cfg.batch_size) == 1:
                    print("Epoch", epoch,
                            ", train step", step_cnt, "/", len(train_loader),
                            ", loss: ", round(float(_loss.data/step_cnt), 5))
                    torch.save(model.state_dict(), cfg.workdir+'/weights/last.pth')

                    # # TODO get the correct evaluate results
                    # beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits.permute(1, 0, 2))
                    # print(beam_scores[0].argmin())
                    # print(logits.size())
                    # print(out_lens[0][beam_scores[0].argmin()], len(data['text'][0]))
                    # print(beam_results[0][beam_scores[0].argmin()][:out_lens[0][beam_scores[0].argmin()]])
                    # for n in beam_results[0][beam_scores[0].argmin()][:out_lens[0][beam_scores[0].argmin()]]:
                    #     # if vocabulary[int(n)] != '<EMP>':
                    #     #     print(vocabulary[n],end = '')
                    #     # else:
                    #     #     break
                    #     print(vocabulary[n],end = '')

                    # print(" ")
                    # for n in data['text'][0]:
                    #     print(vocabulary[int(n)],end = '')
                    # print(" ")
                    
                    # # exit()
                    # # # beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
                    # # zero = torch.zeros_like(beam_results)
                    # # beam_results = torch.where(beam_results > 27, zero, beam_results)
                    # # beam_results = torch.where(beam_results < 0, zero, beam_results)
                    # voc = np.tile(vocabulary, (cfg.batch_size, 1))
                    # pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
                    # text_np = np.take(voc, text[0].data.cpu().numpy().astype(int))

                    # # # print('pred: ', pred.transpose(1, 0))
                    # # print('pred: ')
                    # # for  i, w in enumerate(pred.transpose(1, 0)[0]):
                    # #     if w != '<EMP>':
                    # #         print(w, end="")
                    # #     elif w == '<EMP>':
                    # #         break

                    # # print("")
                    # # print("gt: ")
                    # # for  i, w in enumerate(pred.transpose(1, 0)[0]):
                    # #     if i < 256:
                    # #         print(text_np[0][i], end="")
                    # tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
                    # print('tp: ', tp, 'pred: ', pred, 'pos: ', pos)
                    
                step_cnt += 1
            except:
                continue

        _loss /= len(train_loader)
        writer.add_scalar('train/loss', _loss, epoch)
        train_loss_list.append(float(_loss))
        torch.cuda.empty_cache()

        model = pruning(model, cfg.sparse_mode)
        sparsity = cal_sparsity(model)
        print(sparsity)
        loss_val = validate(val_loader, model, loss_fn)
        writer.add_scalar('val/loss', loss_val, epoch)
        val_loss_list.append(float(loss_val))


        if loss_val < best_loss:
            not_better_cnt = 0
            torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
            print("saved", cfg.workdir+'/weights/best.pth', not_better_cnt)
            best_loss = loss_val
        else:
            not_better_cnt += 1

        if not_better_cnt > 4:
            write_excel(os.path.join(cfg.work_root, cfg.save_excel), 
                            cfg.exp_name, train_loss_list, val_loss_list)
            exit()

def validate(val_loader, model, loss_fn):
    model.eval()
    _loss = 0.0
    step_cnt = 0
    for data in val_loader:
        wave = data['wave'].cuda()  # [1, 128, 109]
        logits = model(wave)
        logits = logits.permute(2, 0, 1)
        logits = F.log_softmax(logits, dim=2)
        if data['text'].size(0) == cfg.batch_size:
            for i in range(cfg.batch_size):
                if i == 0:
                    text = data['text'][i][0:data['length_text'][i]].cuda()
                    # print(data['text'].size())
                    # print(data['length_text'][i])
                else:
                    text = torch.cat([text, 
                                data['text'][i][0: data['length_text'][i]].cuda()])
        else:
            continue
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
    cfg.work_root   = '/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/'
    cfg.workdir     = cfg.work_root + args.exp + '/debug'
    cfg.sparse_mode = args.sparse_mode
    cfg.batch_size  = args.batch_size
    cfg.lr          = args.lr
    cfg.load_from   = args.load_from
    cfg.save_excel   = args.save_excel

    if args.skip_exist == True:
        if os.path.exists(cfg.workdir):
            exit()

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
    train_loader = DataLoader(vctk_train, batch_size=cfg.batch_size, num_workers=8, shuffle=True, pin_memory=True)

    # train_loader = dataset.create("data/v28/train.record", cfg.batch_size, repeat=True)
    vctk_val = VCTK(cfg, 'val')
    val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    # build model
    model = WaveNet(num_classes=28, channels_in=20, dilations=[1,2,4,8,16])
    model = nn.DataParallel(model)
    model.cuda()


    name_list = list()
    para_list = list()
    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    a = model.state_dict()
    for i, name in enumerate(name_list):
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
            raw_w = para_list[i]
            nn.init.xavier_normal_(raw_w, gain=1.0)
            a[name] = raw_w
    model.load_state_dict(a)
    

    weights_dir = os.path.join(cfg.workdir, 'weights')
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    if not os.path.exists(cfg.vis_dir):
        os.mkdir(cfg.vis_dir)
    if args.vis_pattern == True or args.vis_mask == True:
        cfg.vis_dir = os.path.join(cfg.vis_dir, cfg.exp_name)
        if not os.path.exists(cfg.vis_dir):
            os.mkdir(cfg.vis_dir)
    model.train()

    if cfg.resume and os.path.exists(cfg.workdir + '/weights/best.pth'):
        model.load_state_dict(torch.load(cfg.workdir + '/weights/best.pth'), strict=False)
        print("loading", cfg.workdir + '/weights/best.pth')

    if os.path.exists(cfg.load_from):
        model.load_state_dict(torch.load(cfg.load_from), strict=False)
        print("loading", cfg.load_from)

    if args.find_pattern == True:
        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        for i, name in enumerate(name_list):
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                raw_w = para_list[i]
                if raw_w.size(0) == 128 and raw_w.size(1) == 128:
                    patterns = find_pattern_by_similarity(raw_w, 16, [16,16], 0.01, 100)
                    print(patterns)



    if cfg.sparse_mode == 'sparse_pruning':
        cfg.sparsity = args.sparsity
        print(f'sparse_pruning {cfg.sparsity}')

    elif cfg.sparse_mode == 'pattern_pruning':
        print(args.pattern_para)
        pattern_num   = int(args.pattern_para.split('_')[0])
        pattern_shape = [int(args.pattern_para.split('_')[1]), int(args.pattern_para.split('_')[2])]
        pattern_nnz   = int(args.pattern_para.split('_')[3])
        print(f'pattern_pruning {pattern_num} [{pattern_shape[0]}, {pattern_shape[1]}] {pattern_nnz}')
        cfg.patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)
        cfg.pattern_mask = generate_pattern_mask(model, cfg.patterns)

    elif cfg.sparse_mode == 'coo_pruning':
        cfg.coo_shape   = [int(args.coo_para.split('_')[0]), int(args.coo_para.split('_')[1])]
        cfg.coo_nnz   = int(args.coo_para.split('_')[2])
        # cfg.patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)
        print(f'coo_pruning [{cfg.coo_shape[0]}, {cfg.coo_shape[1]}] {cfg.coo_nnz}')
        
    elif cfg.sparse_mode == 'ptcoo_pruning':
        cfg.pattern_num   = int(args.pattern_para.split('_')[0])
        cfg.pattern_shape = [int(args.ptcoo_para.split('_')[1]), int(args.ptcoo_para.split('_')[2])]
        cfg.pt_nnz   = int(args.ptcoo_para.split('_')[3])
        cfg.coo_nnz   = int(args.ptcoo_para.split('_')[4])
        cfg.patterns = generate_pattern(cfg.pattern_num, cfg.pattern_shape, cfg.pt_nnz)
        cfg.pattern_mask = generate_pattern_mask(model, cfg.patterns)
        print(f'ptcoo_pruning {cfg.pattern_num} [{cfg.pattern_shape[0]}, {cfg.pattern_shape[1]}] {cfg.pt_nnz} {cfg.coo_nnz}')


    if args.vis_mask == True:
        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        for i, name in enumerate(name_list):
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                raw_w = para_list[i]

                zero = torch.zeros_like(raw_w)
                one = torch.ones_like(raw_w)

                mask = torch.where(raw_w == 0, zero, one)
                vis.save_visualized_mask(mask, name)
        exit()

    if args.vis_pattern == True:
        pattern_count_dict = find_pattern_model(model, [8,8])
        patterns = list(pattern_count_dict.keys())
        counts = list(pattern_count_dict.values())
        print(len(patterns))
        print(counts)
        vis.save_visualized_pattern(patterns)
        exit()
    # build loss
    loss_fn = nn.CTCLoss(blank=27)
    # loss_fn = nn.CTCLoss()

    #
    scheduler = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(train_step, milestones=[50, 150, 250], gamma=0.5)

    # train
    train(train_loader, scheduler, model, loss_fn, val_loader, writer)
    # val
    # loss = validate(val_loader, scheduler, model, loss_fn)

if __name__ == '__main__':
    main()

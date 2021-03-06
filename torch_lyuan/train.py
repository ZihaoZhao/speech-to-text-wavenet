#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
<<<<<<< HEAD
# LastEditTime : 2021-03-26 09:44:53
=======
# LastEditTime : 2021-02-23 21:10:00
>>>>>>> 88985b5155b13dbfec202bd156e5ba83b471798a
# FilePath     : /speech-to-text-wavenet/torch_lyuan/train.py
# Description  : 0.001 0-5, 0.0001
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
import torch.onnx

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='WaveNet for speech recognition.')
    parser.add_argument('--exp', type=str, help='exp dir', default="default")
    parser.add_argument('--resume', action='store_true', help='resume from exp_name/best.pth', default=False)

    parser.add_argument('--vis_mask', action='store_true', help='visualize and save masks', default=False)
    parser.add_argument('--vis_pattern', action='store_true', help='visualize and save patterns', default=False)

    parser.add_argument('--sparse_mode', type=str, help='dense, sparse_pruning, thre_pruning, pattern_pruning', default="dense")
    parser.add_argument('--sparsity', type=float, help='0.2, 0.4, 0.8', default=0.2)
    parser.add_argument('--pattern_para', type=str, help='[pt_num_pt_shape0_pt_shape1_nnz]', default='16_16_16_128')
    parser.add_argument('--coo_para', type=str, help='[pt_shape0, pt_shape1, nnz]', default='8_8_32')
    parser.add_argument('--ptcoo_para', type=str, help='[pt_num, pt_shape0, pt_shape1, pt_nnz, coo_nnz]', default='16_16_16_128_64')
    parser.add_argument('--find_retrain_para', type=str, help='[pt_num, pt_shape0, pt_shape1, pt_nnz, coo_num, l or m]', default='16_4_4_2_1_m')
    parser.add_argument('--hcgs_para', type=str, help='[block_shape0, block_shape1,reserve_num1, reserve_num2]', default='16, 16, 4, 8')

    parser.add_argument('--batch_size', type=int, help='1, 16, 32', default=16)
    parser.add_argument('--lr', type=float, help='0.001 for tensorflow', default=0.001)

    parser.add_argument('--load_from', type=str, help='.pth', default="/z")
    parser.add_argument('--load_from_h5', type=str, help='.pth', default="/z")

    parser.add_argument('--skip_exist', action='store_true', help='if exist', default=False)
    parser.add_argument('--save_excel', type=str, help='exp.xls', default="default.xls")

    parser.add_argument('--find_pattern', action='store_true', help='find_pattern', default=False)
    parser.add_argument('--find_pattern_para', type=str, help='[zerothre_scorethre]', default='0.02_32')
    parser.add_argument('--find_pattern_shape', type=str, help='[zerothre_scorethre]', default='16_16')
    parser.add_argument('--save_pattern_count_excel', type=str, help='exp.xls', default="pattern_count.xls")

    parser.add_argument('--test_acc', action='store_true', help='test_acc', default=False)
    parser.add_argument('--test_acc_cmodel', action='store_true', help='test_acc_cmodel', default=False)
    parser.add_argument('--test_acc_excel', type=str, help='exp.xls', default="test_acc.xls")

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
        # start_time = time.time()
        _loss = 0.0
        step_cnt = 0
        
        if epoch == 0:
            if cfg.sparse_mode == 'find_retrain':
                print("find_pattern_start")
                cfg.fd_rtn_pattern_set = dict()
                name_list = list()
                para_list = list()
                for name, para in model.named_parameters():
                    name_list.append(name)
                    para_list.append(para)
                    
                cnt = 0


                patterns_dir = os.path.join(cfg.workdir, 'patterns')
                if not os.path.exists(patterns_dir):
                    os.mkdir(patterns_dir)

                weights_dir = os.path.join(cfg.workdir, 'weights_save')
                if not os.path.exists(weights_dir):
                    os.mkdir(weights_dir)

                if cfg.layer_or_model_wise == "l":
                    for i, name in enumerate(name_list):
                        if name.split(".")[-2] != "bn" \
                            and name.split(".")[-2] != "bn2" \
                            and name.split(".")[-2] != "bn3" \
                            and name.split(".")[-1] != "bias":
                            raw_w = para_list[i]
                            print(name, raw_w.size())
                            if raw_w.size(0) == 128 and raw_w.size(1) == 128:
                                cfg.fd_rtn_pattern_set[name], _ = find_top_k_by_kmeans(
                                    raw_w, cfg.pattern_num, cfg.pattern_shape, cfg.pattern_nnz, stride=cfg.pattern_shape)
                            elif raw_w.size(0) == 128 and raw_w.size(1) == 40:
                                # raw_w_pad = torch.cat([raw_w, torch.zeros(raw_w.size(0), 4, raw_w.size(2)).cuda()], 1)
                                cfg.fd_rtn_pattern_set[name], _ = find_top_k_by_kmeans(
                                    raw_w, cfg.pattern_num, cfg.pattern_shape, cfg.pattern_nnz, stride=cfg.pattern_shape)
                            elif raw_w.size(0) == 28 and raw_w.size(1) == 128:
                                raw_w_pad = torch.cat([raw_w, torch.zeros(4, raw_w.size(1), raw_w.size(2)).cuda()], 0)
                                cfg.fd_rtn_pattern_set[name], _ = find_top_k_by_kmeans(
                                    raw_w_pad, cfg.pattern_num, cfg.pattern_shape, cfg.pattern_nnz, stride=cfg.pattern_shape)
                            print(name, cfg.fd_rtn_pattern_set[name].size())
                            pattern_save = np.array(cfg.fd_rtn_pattern_set[name].cpu()).transpose((0, 1, 3, 2))
                            # raw_w_save = np.array(raw_w.cpu().detach())
                            np.savetxt(os.path.join(patterns_dir, name + '.txt'), pattern_save.flatten())
                    #         np.savetxt(os.path.join(weights_dir, name + '.txt'), raw_w_save.transpose((2, 1, 0)).flatten())
                    # exit()
                elif cfg.layer_or_model_wise == "m":
                    for i, name in enumerate(name_list):
                        if name.split(".")[-2] != "bn" \
                            and name.split(".")[-2] != "bn2" \
                            and name.split(".")[-2] != "bn3" \
                            and name.split(".")[-1] != "bias":
                            raw_w = para_list[i]
                            if raw_w.size(0) == 128 and raw_w.size(1) == 128:
                                if cnt == 0:
                                    raw_w_all = raw_w
                                else:
                                    raw_w_all = torch.cat([raw_w_all, raw_w], 2)
                                cnt += 1
                    cfg.fd_rtn_pattern_set['all'], _ = find_top_k_by_kmeans(
                        raw_w_all, cfg.pattern_num, cfg.pattern_shape, cfg.pattern_nnz, stride=cfg.pattern_shape)
                print("find_pattern_end")

        _tp, _pred, _pos = 0, 0, 0
        for data in train_loader:
            # data = prefetcher.next()
            wave = data['wave'].cuda()  # [1, 128, 109]
            if step_cnt % 10 == 0:
                # print("test1")
                model = pruning(model, cfg.sparse_mode)
                # print("test2")
                model.train() 
            if epoch == 0 and step_cnt == 0:
                # print("test3")
                loss_val = validate(val_loader, model, loss_fn)
                writer.add_scalar('val/loss', loss_val, epoch)
                best_loss = loss_val
                not_better_cnt = 0
                torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
                print("saved", cfg.workdir+'/weights/best.pth', not_better_cnt)
                val_loss_list.append(float(loss_val))
                # f1, val_loss, tps, preds, poses = test_acc(val_loader, model, loss_fn)
                # print(f1, val_loss, tps, preds, poses)
                model.train()    
                # print("test4")

            logits = model(wave)
            mask = torch.zeros_like(logits)
            for n in range(len(data['length_wave'])):
                mask[:, :, :data['length_wave'][n]] = 1
            logits *= mask

            logits = logits.permute(2, 0, 1)
            logits = F.log_softmax(logits, dim=2)
            if data['text'].size(0) == cfg.batch_size:
                for i in range(cfg.batch_size):
                    if i == 0:
                        text = data['text'][i][0:data['length_text'][i]].cuda()
                    else:
                        text = torch.cat([text, 
                                    data['text'][i][0: data['length_text'][i]].cuda()])
            else:
                continue


            # try:
            loss = 0.0
            for i in range(logits.size(1)):
                loss += loss_fn(logits[:data['length_wave'][i], i:i+1, :], data['text'][i][0:data['length_text'][i]].cuda(), data['length_wave'][i], data['length_text'][i])
            loss /= logits.size(1)
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data# * float(data['length_text'].float().mean())


            if epoch == 0 and step_cnt == 10:
                writer.add_scalar('train/loss', _loss, epoch)
                train_loss_list.append(float(_loss))

            if step_cnt % int(1200/cfg.batch_size) == 10:
                print("Epoch", epoch,
                        ", train step", step_cnt, "/", len(train_loader),
                        ", loss: ", round(float(_loss.data/step_cnt), 5))
                torch.save(model.state_dict(), cfg.workdir+'/weights/last.pth')

                if float(_loss.data/step_cnt) < 0.7:
                    # TODO get the correct evaluate results
                    for i in range(logits.size(1)):
                        logit = logits[:data['length_wave'][i], i:i+1, :]
                        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logit.permute(1, 0, 2))

                        voc = np.tile(vocabulary, (cfg.batch_size, 1))
                        pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
                        text_np = np.take(voc, data['text'][i][0:data['length_text'][i]].cpu().numpy().astype(int))

                        pred = [pred]
                        text_np = [text_np]
                        # print(utils.cvt_np2string(pred))
                        # print(utils.cvt_np2string(text_np))

                        tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
                        _tp += tp
                        _pred += pred
                        _pos += pos
                    f1 = 2 * _tp / (_pred + _pos + 1e-10)
                
                    print("             Train tp:", _tp, ",pred:", _pred, ",pos:", _pos, ",f1:", f1)
            step_cnt += 1
            # except:
            #     continue
        # print(time.time()-start_time)
        # exit()
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

        model.train()

        if loss_val < best_loss:
            not_better_cnt = 0
            torch.save(model.state_dict(), cfg.workdir+f'/weights/best.pth')
            print("saved", cfg.workdir+f'/weights/best.pth', not_better_cnt)
            best_loss = loss_val
        else:
            not_better_cnt += 1

        if not_better_cnt > 3:
            write_excel(os.path.join(cfg.work_root, cfg.save_excel), 
                            cfg.exp_name, train_loss_list, val_loss_list)
            # exit()

def validate(val_loader, model, loss_fn):    
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
    model.eval()
    _loss = 0.0
    step_cnt = 0
    _tp, _pred, _pos = 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            logits = model(wave)
            logits = logits.permute(2, 0, 1)
            logits = F.log_softmax(logits + 1e-10, dim=2)
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
            loss = 0.0
            for i in range(logits.size(1)):
                loss += loss_fn(logits[:data['length_wave'][i], i:i+1, :], data['text'][i][0:data['length_text'][i]].cuda(), data['length_wave'][i], data['length_text'][i])
            loss /= logits.size(1)
            _loss += loss.data
            # beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits.permute(1, 0, 2))

            # voc = np.tile(vocabulary, (cfg.batch_size, 1))
            # pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
            # text_np = np.take(voc, data['text'][0][0:data['length_text'][0]].cpu().numpy().astype(int))

            # tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
            # _tp += tp
            # _pred += pred
            # _pos += pos
            # f1 = 2 * _tp / (_pred + _pos + 1e-10)
            
            step_cnt += 1
            # if cnt % 10 == 0:
    print("Val step", step_cnt, "/", len(val_loader),
            ", loss: ", round(float(_loss/len(val_loader)), 5))
    # print("Val tp:", _tp, ",pred:", _pred, ",pos:", _pos, ",f1:", f1)

    return _loss/len(val_loader)


def test_acc(val_loader, model, loss_fn):    
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
    model.eval()
    _loss = 0.0
    step_cnt = 0
    tps, preds, poses = 0, 0, 0
    f_cnt = 0
    with torch.no_grad():
        for data in val_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            if 1:
                print(data['wave'].size())
                np.savetxt("/zhzhao/dataset/VCTK/c_model_input_txt/"+str(f_cnt)+".txt", data['wave'].flatten())
                print(f_cnt)
                f_cnt += 1
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
            for i in range(logits.size(1)):
                logit = logits[:data['length_wave'][i], i:i+1, :]
                beam_results, beam_scores, timesteps, out_lens = decoder.decode(logit.permute(1, 0, 2))

                voc = np.tile(vocabulary, (cfg.batch_size, 1))
                pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
                text_np = np.take(voc, data['text'][i][0:data['length_text'][i]].cpu().numpy().astype(int))

                pred = [pred]
                text_np = [text_np]
                # print(utils.cvt_np2string(pred))
                # print(utils.cvt_np2string(text_np))

                tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
                tps += tp
                preds += pred
                poses += pos
            f1 = 2 * tps / (preds + poses + 1e-10)
            
            step_cnt += 1
            # if cnt % 10 == 0:
    print("Val step", step_cnt, "/", len(val_loader),
            ", loss: ", round(float(_loss.data/step_cnt), 5))
    print("Val tps:", tps, ",preds:", preds, ",poses:", poses, ",f1:", f1)

    return f1, _loss/len(val_loader), tps, preds, poses


def test_acc_cmodel(val_loader, model, loss_fn):    
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
    model.eval()
    _loss = 0.0
    _loss2 = 0.0
    step_cnt = 0
    tps, preds, poses = 0, 0, 0
    tps2, preds2, poses2 = 0, 0, 0
    f_cnt = 0
    with torch.no_grad():
        for data in val_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            # if 1:
            #     print(data['wave'].size())
            #     np.savetxt("/zhzhao/dataset/VCTK/c_model_input_txt/"+str(f_cnt)+".txt", data['wave'].flatten())
            #     print(f_cnt)
            logits = model(wave)
            logits_cmodel = torch.from_numpy(np.loadtxt("/zhzhao/dataset/VCTK/c_model_output_txt/"+str(f_cnt)+".txt").reshape((1, 28, 720)))
            # print(logits_cmodel.reshape((28, 1, 720)))
            f_cnt += 1
            logits = logits.permute(2, 0, 1)
            logits_cmodel = logits_cmodel.permute(2, 0, 1)
            print(logits)
            print(logits_cmodel)
            # logits_cmodel = logits_cmodel.permute(2, 0, 1)
            logits = F.log_softmax(logits, dim=2)
            logits_cmodel = F.log_softmax(logits_cmodel, dim=2)
            if data['text'].size(0) == 1:
                for i in range(1):
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
            loss2 = loss_fn(logits_cmodel, text, data['length_wave'], data['length_text'])
            _loss += loss.data
            _loss2 += loss2.data
            # print(_loss)
            # print(_loss2)
            for i in range(logits.size(1)):
                logit = logits[:data['length_wave'][i], i:i+1, :]
                beam_results, beam_scores, timesteps, out_lens = decoder.decode(logit.permute(1, 0, 2))

                voc = np.tile(vocabulary, (1, 1))
                pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
                text_np = np.take(voc, data['text'][i][0:data['length_text'][i]].cpu().numpy().astype(int))

                pred = [pred]
                text_np = [text_np]
                # print(utils.cvt_np2string(pred))
                # print(utils.cvt_np2string(text_np))

                tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
                tps += tp
                preds += pred
                poses += pos
            f1 = 2 * tps / (preds + poses + 1e-10)

            for i in range(logits_cmodel.size(1)):
                logit = logits_cmodel[:data['length_wave'][i], i:i+1, :]
                beam_results, beam_scores, timesteps, out_lens = decoder.decode(logit.permute(1, 0, 2))

                voc = np.tile(vocabulary, (1, 1))
                pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
                text_np = np.take(voc, data['text'][i][0:data['length_text'][i]].cpu().numpy().astype(int))

                pred = [pred]
                text_np = [text_np]
                # print(utils.cvt_np2string(pred))
                # print(utils.cvt_np2string(text_np))

                tp2, pred2, pos2 = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
                tps2 += tp2
                preds2 += pred2
                poses2 += pos2
            f12 = 2 * tps2 / (preds2 + poses2 + 1e-10)
            
            step_cnt += 1
            print("Val step", step_cnt, "/", len(val_loader),
                    ", loss: ", round(float(_loss.data/step_cnt), 5))
            print("Val tps:", tps, ",preds:", preds, ",poses:", poses, ",f1:", f1)
            print("C Model Val step", step_cnt, "/", len(val_loader),
                    ", loss: ", round(float(_loss2.data/step_cnt), 5))
            print("C Model Val tps:", tps2, ",preds:", preds2, ",poses:", poses2, ",f1:", f12)
            print(" ")
            # if f_cnt > 6 :
            #     break

    return f1, _loss/len(val_loader), tps, preds, poses



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
    
    if args.find_pattern == True:
        cfg.find_pattern_num   = 16
        cfg.find_pattern_shape = [int(args.find_pattern_shape.split('_')[0]), int(args.find_pattern_shape.split('_')[1])]
        cfg.find_zero_threshold = float(args.find_pattern_para.split('_')[0])
        cfg.find_score_threshold = int(args.find_pattern_para.split('_')[1])
        if int(cfg.find_pattern_shape[0] * cfg.find_pattern_shape[1]) <= cfg.find_score_threshold:
            exit()

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
    train_loader = DataLoader(vctk_train, batch_size=cfg.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    # train_loader = dataset.create("data/v28/train.record", cfg.batch_size, repeat=True)
    vctk_val = VCTK(cfg, 'val')
    if args.test_acc_cmodel == True:
        val_loader = DataLoader(vctk_val, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    else:
        val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    # build model
    model = WaveNet(num_classes=28, channels_in=40, dilations=[1,2,4,8,16])
    model = nn.DataParallel(model)
    model.cuda()


    name_list = list()
    para_list = list()
    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    a = model.state_dict()
    for i, name in enumerate(name_list):
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
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
        model.load_state_dict(torch.load(cfg.workdir + '/weights/best.pth'), strict=True)
        print("loading", cfg.workdir + '/weights/best.pth')
        cfg.load_from = cfg.workdir + '/weights/best.pth'

    if args.test_acc == True:
        if os.path.exists(cfg.load_from):
            model.load_state_dict(torch.load(cfg.load_from), strict=True)
            print("loading", cfg.load_from)
        else:
            print("Error: model file not exists, ", cfg.load_from)
            exit()
    else:
        if os.path.exists(cfg.load_from):
            model.load_state_dict(torch.load(cfg.load_from), strict=True)
            print("loading", cfg.load_from)
            # Export the model
            print("exporting onnx ...")
            model.eval()
            batch_size = 1
            x = torch.randn(batch_size, 40, 720, requires_grad=True).cuda()
            torch.onnx.export(model.module,               # model being run
                            x,                         # model input (or a tuple for multiple inputs)
                            "wavenet.onnx",   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=10,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                            'output' : {0 : 'batch_size'}})

    if os.path.exists(args.load_from_h5):
        # model.load_state_dict(torch.load(args.load_from_h5), strict=True)
        print("loading", args.load_from_h5)
        model.train()
        model_dict = model.state_dict()
        print(model_dict.keys())
        #先将参数值numpy转换为tensor形式
        pretrained_dict = dd.io.load(args.load_from_h5)
        print(pretrained_dict.keys())
        new_pre_dict = {}
        for k,v in pretrained_dict.items():
            new_pre_dict[k] = torch.Tensor(v)
        #更新
        model_dict.update(new_pre_dict)
        #加载
        model.load_state_dict(model_dict)

    if args.find_pattern == True:

        # cfg.find_pattern_num   = 16
        # cfg.find_pattern_shape = [int(args.find_pattern_shape.split('_')[0]), int(args.find_pattern_shape.split('_')[1])]
        # cfg.find_zero_threshold = float(args.find_pattern_para.split('_')[0])
        # cfg.find_score_threshold = int(args.find_pattern_para.split('_')[1])

        # if cfg.find_pattern_shape[0] * cfg.find_pattern_shape[0] <= cfg.find_score_threshold:
        #     exit()

        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        for i, name in enumerate(name_list):
            if name.split(".")[-2] != "bn" \
                and name.split(".")[-2] != "bn2" \
                and name.split(".")[-2] != "bn3" \
                and name.split(".")[-1] != "bias":
                raw_w = para_list[i]
                if raw_w.size(0) == 128 and raw_w.size(1) == 128:
                    patterns, pattern_match_num, pattern_coo_nnz, pattern_nnz, pattern_inner_nnz \
                                    = find_pattern_by_similarity(raw_w
                                        , cfg.find_pattern_num
                                        , cfg.find_pattern_shape
                                        , cfg.find_zero_threshold
                                        , cfg.find_score_threshold)

                    pattern_num_memory_dict, pattern_num_cal_num_dict, pattern_num_coo_nnz_dict \
                                    = pattern_curve_analyse(raw_w.shape
                                        , cfg.find_pattern_shape
                                        , patterns
                                        , pattern_match_num
                                        , pattern_coo_nnz
                                        , pattern_nnz
                                        , pattern_inner_nnz)
                                        
                    write_pattern_curve_analyse(os.path.join(cfg.work_root, args.save_pattern_count_excel)
                                        , cfg.exp_name + " " + args.find_pattern_shape + " " + args.find_pattern_para
                                        , patterns, pattern_match_num, pattern_coo_nnz, pattern_nnz
                                        , pattern_inner_nnz
                                        , pattern_num_memory_dict, pattern_num_cal_num_dict, pattern_num_coo_nnz_dict)

                    # write_pattern_count(os.path.join(cfg.work_root, args.save_pattern_count_excel)
                    #                     , cfg.exp_name + " " + args.find_pattern_shape +" " + args.find_pattern_para
                    #                     , all_nnzs.values(), all_patterns.values())
                    exit()



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
        
    elif cfg.sparse_mode == 'find_retrain':
        cfg.pattern_num   = int(args.find_retrain_para.split('_')[0])
        cfg.pattern_shape = [int(args.find_retrain_para.split('_')[1]), int(args.find_retrain_para.split('_')[2])]
        cfg.pattern_nnz   = int(args.find_retrain_para.split('_')[3])
        cfg.coo_num       = float(args.find_retrain_para.split('_')[4])
        cfg.layer_or_model_wise   = str(args.find_retrain_para.split('_')[5])
        # cfg.fd_rtn_pattern_candidates = generate_complete_pattern_set(
        #                                 cfg.pattern_shape, cfg.pattern_nnz)
        print(f'find_retrain {cfg.pattern_num} [{cfg.pattern_shape[0]}, {cfg.pattern_shape[1]}] {cfg.pattern_nnz} {cfg.coo_num} {cfg.layer_or_model_wise}')

    elif cfg.sparse_mode == 'hcgs_pruning':
        print(args.pattern_para)
        cfg.block_shape = [int(args.hcgs_para.split('_')[0]), int(args.hcgs_para.split('_')[1])]
        cfg.reserve_num1 = int(args.hcgs_para.split('_')[2])
        cfg.reserve_num2 = int(args.hcgs_para.split('_')[3])
        print(f'hcgs_pruning {cfg.reserve_num1}/8 {cfg.reserve_num2}/16')
        cfg.hcgs_mask = generate_hcgs_mask(model, cfg.block_shape, cfg.reserve_num1, cfg.reserve_num2)

    if args.vis_mask == True:
        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        for i, name in enumerate(name_list):
            if name.split(".")[-2] != "bn" \
                and name.split(".")[-2] != "bn2" \
                and name.split(".")[-2] != "bn3" \
                and name.split(".")[-1] != "bias":
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
    if args.test_acc == True:
        f1, val_loss, tps, preds, poses = test_acc(val_loader, model, loss_fn)
        # f1, val_loss, tps, preds, poses = test_acc(val_loader, model, loss_fn)
        write_test_acc(os.path.join(cfg.work_root, args.test_acc_excel), 
                        cfg.exp_name, f1, val_loss, tps, preds, poses)
        exit()

    if args.test_acc_cmodel == True:
        f1, val_loss, tps, preds, poses = test_acc_cmodel(val_loader, model, loss_fn)
        # f1, val_loss, tps, preds, poses = test_acc(val_loader, model, loss_fn)
        write_test_acc(os.path.join(cfg.work_root, args.test_acc_excel), 
                        cfg.exp_name, f1, val_loss, tps, preds, poses)
        exit()
    # train
    train(train_loader, scheduler, model, loss_fn, val_loader, writer)
    # val
    # loss = validate(val_loader, scheduler, model, loss_fn)

if __name__ == '__main__':
    main()

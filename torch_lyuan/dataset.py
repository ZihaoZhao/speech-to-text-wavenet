#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-11 15:28:41
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-15 08:55:59
# FilePath     : /speech-to-text-wavenet/torch_lyuan/dataset.py
# Description  : 
#-------------------------------------------# 

import torch
from torch.utils.data import Dataset
import utils
import random
import json
import os
import numpy as np



class VCTK(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['train', 'val']
        if not os.path.exists(self.cfg.datalist):
            raise ValueError('datalist must exists, initial datalist is not supported')
        self.train_filenames, self.test_filenames = json.load(open(self.cfg.datalist, 'r', encoding='utf-8'))
        if self.mode == 'train':
            self.max_wave = 520
            self.max_text = 256
        else:
            self.max_wave = 720
            self.max_text = 256

    def __getitem__(self, idx):
        if self.mode =='train':
            filenames = self.train_filenames[idx]
        else:
            filenames = self.test_filenames[idx]
        wave_path = self.cfg.dataset + filenames[0]
        txt_path = self.cfg.dataset + filenames[1]
        wave_tmp = utils.read_wave(wave_path) # numpy
        wave_tmp = torch.from_numpy(wave_tmp)
        wave = torch.zeros([20,self.max_wave]) # 512 may be too short, if error,fix it
        length_wave = wave_tmp.shape[1]
        wave[:,:length_wave] = wave_tmp

        text_tmp = utils.read_txt(txt_path)  # list
        length_text = len(text_tmp)
        text_tmp = torch.tensor(text_tmp)
        text = torch.zeros([self.max_text]) # 256 may be too short, fix it, if error
        text[:length_text] = text_tmp
        name = filenames[0].split('/')[-1]

        if length_text >= length_wave:
            sample = {'name':name, 'wave':torch.zeros([20, self.max_wave],dtype=torch.float), 'text':torch.zeros([self.max_text],dtype=torch.float),
                    'length_wave':self.max_wave, 'length_text':self.max_text}
        else:
            sample = {'name':name, 'wave':wave, 'text':text,
                    'length_wave':length_wave, 'length_text':length_text}
        return sample


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_filenames)
        else:
            return len(self.test_filenames)

if __name__ == '__main__':
    # train_filenames, test_filenames = json.load(open('/lyuan/code/speech-to-text-wavenet/data/list.json', 'r', encoding='utf-8'))
    # print(len(train_filenames), train_filenames) #[['/VCTK-Corpus/wav48/p376/p376_076.wav', '/VCTK-Corpus/txt/p376/p376_076.txt'], ['/VCTK-Corpus/wav48/p376/p376_021.wav', '/VCTK-Corpus/txt/p376/p376_021.txt']]
    import config_train as cfg
    # vctk = VCTK(cfg, 'train')
    # length = len(vctk)
    # max_length = 0
    # for i in range(length):
    #     tmp = vctk[i]['wave'].shape[1]
    #     if tmp>max_length:
    #         max_length = tmp
    # print(f'train set {max_length}')
    vctk = VCTK(cfg, 'val')
    length = len(vctk)
    max_wave = 0
    max_text = 0
    for i in range(length):
        tmp = vctk[i]['length_wave']
        if tmp>max_wave:
            max_length = tmp
        tmp = vctk[i]['length_text']
        if tmp > max_text:
            max_length = tmp
    print(f'val set {max_wave}, {max_text}')


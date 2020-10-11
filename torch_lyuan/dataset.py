#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-11 10:14:04
# FilePath     : /speech-to-text-wavenet/torch_lyuan/dataset.py
# Description  : 
#-------------------------------------------# 

import torch
from torch.utils.data import Dataset
import utils
import random
import json
import os


class VCTK(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['train', 'val']
        if not os.path.exists(self.cfg.datalist):
            raise ValueError('datalist must exists, initial datalist is not supported')
        self.train_filenames, self.test_filenames = json.load(open(self.cfg.datalist, 'r', encoding='utf-8'))


    def __getitem__(self, idx):
        if self.mode =='train':
            filenames = self.train_filenames[idx]
        else:
            filenames = self.test_filenames[idx]
        wave_path = self.cfg.dataset + filenames[0]
        txt_path = self.cfg.dataset + filenames[1]
        wave = utils.read_wave(wave_path) # numpy
        text = utils.read_txt(txt_path) # list
        wave = torch.from_numpy(wave)
        text = torch.tensor(text)
        name = filenames[0].split('/')[-1]
        sample = {'name':name, 'wave':wave, 'text':text}
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
    vctk = VCTK(cfg, 'train')
    print(len(vctk))
    print(vctk[3]['wave'].shape)
    print(vctk[3]['text'].shape)


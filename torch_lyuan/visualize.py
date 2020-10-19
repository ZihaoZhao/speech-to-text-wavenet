#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-19 16:38:18
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-19 16:42:31
# FilePath     : /speech-to-text-wavenet/torch_lyuan/visualize.py
# Description  : 
#-------------------------------------------# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import cv2
import imageio
import os


def visualize(input):
    plt.matshow(input, cmap='hot')
    plt.colorbar()
    plt.show()


def save_visualize(input, image_name):
    plt.matshow(input, cmap='hot')
    # plt.matshow(input_image, cmap='hot', vmin = 0, vmax = 1)
    # plt.colorbar()
    plt.savefig(image_name, dpi=300)
        
def save_pattern(model):


if __name__ == '__main__':
    model = WaveNet(num_classes=28, channels_in=20, dilations=[1,2,4,8,16])
    model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load(cfg.workdir + '/weights/best.pth'))
    save_pattern(model)
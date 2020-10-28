###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-28 14:24:15
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train_script/find_pattern.sh
 # @Description  : 
### 
#!/usr/bin/env bash

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/find_pattern.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.01_64
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.02_64
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.04_64
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.01_32
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.02_32
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.04_32
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.01_16
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.02_16
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.04_16


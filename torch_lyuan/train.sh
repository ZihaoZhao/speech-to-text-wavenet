###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-23 07:43:59
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train.sh
 # @Description  : 
### 
#!/usr/bin/env bash


#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_128 --sparse_mode pattern_pruning --pattern_para 16_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_192 --sparse_mode pattern_pruning --pattern_para 16_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_64 --sparse_mode pattern_pruning --pattern_para 16_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_128_64 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_128_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_85_43 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_80_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_43_21 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_32_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --sparse_mode coo_pruning --coo_para 8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --sparse_mode coo_pruning --coo_para 8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --sparse_mode coo_pruning --coo_para 8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
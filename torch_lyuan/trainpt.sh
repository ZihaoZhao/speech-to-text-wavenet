###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-23 07:48:10
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/trainpt.sh
 # @Description  : 
### 
#!/usr/bin/env bash


#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/trainpt.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_1616_192 --sparse_mode pattern_pruning --pattern_para 8_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_1616_128 --sparse_mode pattern_pruning --pattern_para 8_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_1616_64 --sparse_mode pattern_pruning --pattern_para 8_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_1616_192 --sparse_mode pattern_pruning --pattern_para 4_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_1616_128 --sparse_mode pattern_pruning --pattern_para 4_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_1616_64 --sparse_mode pattern_pruning --pattern_para 4_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_1616_192 --sparse_mode pattern_pruning --pattern_para 2_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_1616_128 --sparse_mode pattern_pruning --pattern_para 2_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_1616_64 --sparse_mode pattern_pruning --pattern_para 2_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_1616_192 --sparse_mode pattern_pruning --pattern_para 1_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_1616_128 --sparse_mode pattern_pruning --pattern_para 1_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth 
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_1616_64 --sparse_mode pattern_pruning --pattern_para 1_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  
###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-23 20:30:44
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train0_0001.sh
 # @Description  : 
### 
#!/usr/bin/env bash


#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_192 --sparse_mode pattern_pruning --pattern_para 16_16_16_192_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_128 --sparse_mode pattern_pruning --pattern_para 16_16_16_128_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth  --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_64 --sparse_mode pattern_pruning --pattern_para 16_16_16_64_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth   --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_128_64 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_128_64_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth   --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_85_43 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_80_48_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth   --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_43_21 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_32_32_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth   --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --sparse_mode coo_pruning --coo_para 8_8_48_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth   --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --sparse_mode coo_pruning --coo_para 8_8_32_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth   --lr 0.0001
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --sparse_mode coo_pruning --coo_para 8_8_16_0_0001 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth   --lr 0.0001
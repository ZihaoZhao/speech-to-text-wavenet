###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-11-02 22:26:16
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/script/train_coo16.sh
 # @Description  : 
### 
#!/usr/bin/env bash


#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/script/train_coo16.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_256 --sparse_mode coo_pruning --coo_para 16_16_256 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_224 --sparse_mode coo_pruning --coo_para 16_16_224 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_192 --sparse_mode coo_pruning --coo_para 16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_160 --sparse_mode coo_pruning --coo_para 16_16_160 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_128 --sparse_mode coo_pruning --coo_para 16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_96  --sparse_mode coo_pruning --coo_para 16_16_96  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_64  --sparse_mode coo_pruning --coo_para 16_16_64  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_16_16_32  --sparse_mode coo_pruning --coo_para 16_16_32  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo16.xls
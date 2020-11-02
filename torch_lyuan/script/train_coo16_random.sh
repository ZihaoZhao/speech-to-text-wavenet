###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-26 19:22:57
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train_script/train_coo16_random.sh
 # @Description  : 
### 
#!/usr/bin/env bash


#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/train_coo.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_256 --sparse_mode coo_pruning --coo_para 16_16_256 --lr 0.001  --skip_exist  --save_excel random_coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_224 --sparse_mode coo_pruning --coo_para 16_16_224 --lr 0.001  --skip_exist  --save_excel random_coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_192 --sparse_mode coo_pruning --coo_para 16_16_192 --lr 0.001  --skip_exist  --save_excel random_coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_160 --sparse_mode coo_pruning --coo_para 16_16_160 --lr 0.001  --skip_exist  --save_excel random_coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_128 --sparse_mode coo_pruning --coo_para 16_16_128 --lr 0.001  --skip_exist  --save_excel random_coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_96  --sparse_mode coo_pruning --coo_para 16_16_96  --lr 0.001  --skip_exist  --save_excel random_coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_64  --sparse_mode coo_pruning --coo_para 16_16_64  --lr 0.001  --skip_exist  --save_excel random_coo16.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_coo_16_16_32  --sparse_mode coo_pruning --coo_para 16_16_32  --lr 0.001  --skip_exist  --save_excel random_coo16.xls
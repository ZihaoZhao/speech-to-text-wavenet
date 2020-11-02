###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-10-24 09:12:30
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train.sh
 # @Description  : 
### 
#!/usr/bin/env bash


#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/train_coo.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_64 --sparse_mode coo_pruning --coo_para 8_8_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_56 --sparse_mode coo_pruning --coo_para 8_8_56 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --sparse_mode coo_pruning --coo_para 8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_40 --sparse_mode coo_pruning --coo_para 8_8_40 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --sparse_mode coo_pruning --coo_para 8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_24 --sparse_mode coo_pruning --coo_para 8_8_24 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --sparse_mode coo_pruning --coo_para 8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_8  --sparse_mode coo_pruning --coo_para 8_8_8  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel coo.xls
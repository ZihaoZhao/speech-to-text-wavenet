###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-26 19:27:13
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train_script/train_ptcoo_random.sh
 # @Description  : 
### 
#!/usr/bin/env bash

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/train_ptcoo.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_192_64 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_192_64 --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_168_56 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_168_56 --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_144_48 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_144_48 --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_120_40 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_120_40 --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_96_32  --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_96_32  --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_72_24  --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_72_24  --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_48_16  --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_48_16  --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp random_ptcoo_16_16_16_24_8   --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_24_8   --lr 0.001   --skip_exist  --save_excel random_ptcoo.xls
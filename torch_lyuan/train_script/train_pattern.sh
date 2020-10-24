###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-10-24 09:09:28
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train.sh
 # @Description  : 
### 
#!/usr/bin/env bash


#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/train_pattern.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_256 --sparse_mode pattern_pruning --pattern_para 16_16_16_256 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_256 --sparse_mode pattern_pruning  --pattern_para 8_16_16_256  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_256 --sparse_mode pattern_pruning  --pattern_para 4_16_16_256  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_256 --sparse_mode pattern_pruning  --pattern_para 2_16_16_256  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_256 --sparse_mode pattern_pruning  --pattern_para 1_16_16_256  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_224 --sparse_mode pattern_pruning --pattern_para 16_16_16_224 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_224 --sparse_mode pattern_pruning  --pattern_para 8_16_16_224  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_224 --sparse_mode pattern_pruning  --pattern_para 4_16_16_224  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_224 --sparse_mode pattern_pruning  --pattern_para 2_16_16_224  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_224 --sparse_mode pattern_pruning  --pattern_para 1_16_16_224  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_192 --sparse_mode pattern_pruning --pattern_para 16_16_16_192--load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_192 --sparse_mode pattern_pruning  --pattern_para 8_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_192 --sparse_mode pattern_pruning  --pattern_para 4_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_192 --sparse_mode pattern_pruning  --pattern_para 2_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_192 --sparse_mode pattern_pruning  --pattern_para 1_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_160 --sparse_mode pattern_pruning --pattern_para 16_16_16_160 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_160 --sparse_mode pattern_pruning  --pattern_para 8_16_16_160  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_160 --sparse_mode pattern_pruning  --pattern_para 4_16_16_160  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_160 --sparse_mode pattern_pruning  --pattern_para 2_16_16_160  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_160 --sparse_mode pattern_pruning  --pattern_para 1_16_16_160  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_128 --sparse_mode pattern_pruning --pattern_para 16_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_128 --sparse_mode pattern_pruning  --pattern_para 8_16_16_128  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_128 --sparse_mode pattern_pruning  --pattern_para 4_16_16_128  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_128 --sparse_mode pattern_pruning  --pattern_para 2_16_16_128  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_128 --sparse_mode pattern_pruning  --pattern_para 1_16_16_128  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_96 --sparse_mode pattern_pruning  --pattern_para 16_16_16_96 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_96 --sparse_mode pattern_pruning   --pattern_para 8_16_16_96 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_96 --sparse_mode pattern_pruning   --pattern_para 4_16_16_96 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_96 --sparse_mode pattern_pruning   --pattern_para 2_16_16_96 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_96 --sparse_mode pattern_pruning   --pattern_para 1_16_16_96 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_64 --sparse_mode pattern_pruning  --pattern_para 16_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_64 --sparse_mode pattern_pruning   --pattern_para 8_16_16_64 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_64 --sparse_mode pattern_pruning   --pattern_para 4_16_16_64 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_64 --sparse_mode pattern_pruning   --pattern_para 2_16_16_64 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_64 --sparse_mode pattern_pruning   --pattern_para 1_16_16_64 ---load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001   --skip_exist  --save_excel pattern.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_16_16_32 --sparse_mode pattern_pruning  --pattern_para 16_16_16_32  --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_16_16_32 --sparse_mode pattern_pruning   --pattern_para 8_16_16_32 - --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_16_16_32 --sparse_mode pattern_pruning   --pattern_para 4_16_16_32 - --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_2_16_16_32 --sparse_mode pattern_pruning   --pattern_para 2_16_16_32 - --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_16_16_32 --sparse_mode pattern_pruning   --pattern_para 1_16_16_32 - --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --lr 0.001  --skip_exist  --save_excel pattern.xls

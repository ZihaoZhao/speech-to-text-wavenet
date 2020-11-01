###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-31 12:01:37
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train_script/find_pattern_coo.sh
 # @Description  : 
### 
#!/usr/bin/env bash

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/find_pattern.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_56 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_56 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_48 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_56 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_56 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_24 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_56 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel coo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_48 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_24 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel coo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_40 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_40 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_48 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_40 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_40 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_24 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_40 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel coo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_32/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_32/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_48 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_32/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_32/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_24 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_32/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel coo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_24 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_24/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_24 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_24/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_48 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_24 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_24/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_24 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_24/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_24 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_24 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_24/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel coo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_16/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_16/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_48 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_16/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_16/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_24 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_16/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel coo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_8 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_8/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_8 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_8/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_48 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_8 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_8/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_8 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_8/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_24 --save_pattern_count_excel coo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_8 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/coo_8_8_8/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel coo_pattern_count.xls


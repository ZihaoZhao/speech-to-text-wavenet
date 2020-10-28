###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-10-28 19:59:21
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train_script/find_pattern_ptcoo.sh
 # @Description  : 
### 
#!/usr/bin/env bash

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/find_pattern.sh

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_168_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_168_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_168_56/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel ptcoo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_144_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_144_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_144_48/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel ptcoo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_120_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_120_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_120_40/debug/weights/best.pth --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel ptcoo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_96_32/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_96_32/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_96_32/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel ptcoo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_72_24/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_72_24/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_72_24/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel ptcoo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_48_16/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_48_16/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_48_16/debug/weights/best.pth  --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel ptcoo_pattern_count.xls

/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_24_8/debug/weights/best.pth   --find_pattern --find_pattern_para 0.001_64 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_24_8/debug/weights/best.pth   --find_pattern --find_pattern_para 0.001_32 --save_pattern_count_excel ptcoo_pattern_count.xls
/zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/ptcoo_16_16_16_24_8/debug/weights/best.pth   --find_pattern --find_pattern_para 0.001_16 --save_pattern_count_excel ptcoo_pattern_count.xls


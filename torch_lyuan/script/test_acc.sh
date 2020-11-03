###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-11-02 20:12:48
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/script/test_acc.sh
 # @Description  : 
### 
#!/usr/bin/env bash

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/script/test_acc.sh

excel_name="test_acc2.xls"
filenames=$(ls /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/)
# for file in ${filenames};do
#     echo ${file}
# done


for file in ${filenames}
do
    if [[ ${file: 0-4} != ".xls"]] && [[${file: 0-4} != ".pth" ]]
    then
        /zhzhao/miniconda3/envs/pytorch16/bin/python \
        /zhzhao/code/wavenet_torch/torch_lyuan/train.py \
        --exp $file \
        --load_from "/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/"$file"/debug/weights/best.pth" \
        --test_acc \
        --test_acc_excel $excel_name
    fi
done


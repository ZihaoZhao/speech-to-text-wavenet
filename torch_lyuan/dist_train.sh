#!/usr/bin/env bash

/lyuan/miniconda3/envs/wavenet/bin/python \-m torch.distributed.launch \--nproc_per_node=1  \--master_port=12767 /lyuan/code/speech-to-text-wavenet/torch_lyuan/train_distributed.py

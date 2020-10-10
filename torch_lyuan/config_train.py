mode = None # set at dataloader, must be None

workdir = '/lyuan/code/speech-to-text-wavenet/torch_lyuan/exp/debug' # all output you need will be saved here, include ckpt.pth.

# path setting
dataset = '/lyuan/dataset/VCTK'
datalist = '/lyuan/code/speech-to-text-wavenet/data/list.json'

batch_size = 1
load_from = ''
epochs = 1000
lr = 1e-5
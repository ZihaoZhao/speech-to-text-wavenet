#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
<<<<<<< HEAD
# LastEditTime : 2021-03-26 09:42:51
=======
# LastEditTime : 2021-02-25 10:19:38
>>>>>>> 88985b5155b13dbfec202bd156e5ba83b471798a
# FilePath     : /speech-to-text-wavenet/torch_lyuan/wavenet.py
# Description  : 
#-------------------------------------------# 

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

import config_train as cfg
'''
there mat be some error in padding, I don't known how padding in slim,
so the skip connection in resnet and the init parameter, dilation, in wavenet class may be error.
'''


def _weights_initializer():
    pass

class Aconv1d(nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(Aconv1d, self).__init__()

        assert activate in ['sigmoid', 'tanh']

        self.dilation = dilation
        self.activate = activate

        self.dilation_conv1d = nn.Conv1d(in_channels=channel_in, out_channels=channel_out,
                                       kernel_size=7, dilation=self.dilation, bias=False)
        self.bn = nn.BatchNorm1d(channel_out)


    def forward(self, inputs):
        # padding number = (kernel_size - 1) * dilation / 2
        # print(inputs.size())
        # print("input:", inputs)
        inputs = F.pad(inputs, (3*self.dilation, 3*self.dilation))
        # print(inputs.size())
        # inputs = F.pad(inputs, (6*self.dilation, 0))


        # print("input:", inputs.size())
        # print("input:", inputs)
        # print(self.dilation_conv1d.weight.shape)
        # print(self.dilation_conv1d.weight.permute((2, 1, 0))[0][0])
        outputs = self.dilation_conv1d(inputs)
        # print("raw:", outputs)
        outputs = self.bn(outputs)
        # print("bn:", outputs)

        if self.activate=='sigmoid':
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.tanh(outputs)

        # print("act:", outputs)
        # exit()
        return outputs


class ResnetBlock(nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(ResnetBlock, self).__init__()
        self.conv_filter = Aconv1d(dilation, channel_in, channel_out, activate='tanh')
        self.conv_gate = Aconv1d(dilation, channel_in, channel_out, activate='sigmoid')

        self.conv1d = nn.Conv1d(channel_out, out_channels=128, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, inputs):
        out_filter = self.conv_filter(inputs)
        # print("filter:", out_filter)
        out_gate = self.conv_gate(inputs)
        # print("gate:", out_gate)
        outputs = out_filter * out_gate

        outputs = torch.tanh(self.bn(self.conv1d(outputs)))
        # print("out:", outputs)
        # exit()
        out = outputs + inputs
        return out, outputs

class WaveNet(nn.Module):
    def __init__(self, num_classes, channels_in, channels_out=128, num_layers=3, dilations=[1,2,4,8,16]): # dilations=[1,2,4]
        super(WaveNet, self).__init__()
        self.num_layers = num_layers
        self.conv1d = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels_out)

        self.resnet_block_0 = nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_1 = nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_2 = nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.conv1d_out = nn.Conv1d(channels_out, channels_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.get_logits = nn.Conv1d(in_channels=channels_out, out_channels=num_classes, kernel_size=1)
        # self.bn3 = nn.BatchNorm1d(num_classes)

    def forward(self, inputs):
        # print("input:", inputs.size())
        # print("input:", inputs)
<<<<<<< HEAD
        # # print(inputs[0][1])
=======
        # print(inputs[0][1])
>>>>>>> 88985b5155b13dbfec202bd156e5ba83b471798a
        # print("out:", self.conv1d(inputs).size())
        # print("out:", self.conv1d(inputs))
        x = self.bn(self.conv1d(inputs))
        # print("after BN", x.size())
        # print("after BN", x)
        x = torch.tanh(x)
        # print("after tanh", x.size())
        # print("after tanh", x)
        # exit()
        # print(self.conv1d.weight.size())
        # print(self.conv1d.weight.permute((2,1,0)))
        # print(self.conv1d.weight.permute((2,0,1))[0][0])
        # print(self.conv1d.weight.permute((2,1,0))[0][0])
        # exit()
        outs = 0.0
        # for _ in range(self.num_layers):
        for layer in self.resnet_block_0:
            # print("block input:", x)
            x, out = layer(x)
            # print(out)
            outs += out
        # exit()
        for layer in self.resnet_block_1:
            x, out = layer(x)
            outs += out
        for layer in self.resnet_block_2:
            x, out = layer(x)
            outs += out

        outs = torch.tanh(self.bn2(self.conv1d_out(outs)))
        # print(outs)
        logits = self.get_logits(outs)
        # print(logits)

        # logits = self.get_logits(outs)

        # exit()

        return logits




if __name__ == '__main__':
    model = WaveNet(num_classes=27, channels_in=40)
    model.eval()
    input = torch.rand([4,16,128]) # [4,16,128] may be too short. maybe there is some error in padding.
    print(model(input))




import torch
from torch import nn

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

        self.activate = activate

        self.dilation_conv1d = nn.Conv1d(in_channels=channel_in, out_channels=channel_out,
                                       kernel_size=7, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channel_out)


    def forward(self, inputs):
        outputs = self.dilation_conv1d(inputs)
        outputs = self.bn(outputs)
        if self.activate=='sigmoid':
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.tanh(outputs)
        return outputs


class ResnetBlock(nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(ResnetBlock, self).__init__()
        self.conv_filter = Aconv1d(dilation, channel_in, channel_out, activate='tanh')
        self.conv_gate = Aconv1d(dilation, channel_in, channel_out, activate='sigmoid')

        self.conv1d = nn.Conv1d(channel_out, out_channels=128, kernel_size=1)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, inputs):
        out_filter = self.conv_filter(inputs)
        out_gate = self.conv_gate(inputs)
        outputs = out_filter * out_gate

        outputs = torch.tanh(self.bn(self.conv1d(outputs)))
        print(outputs.shape, inputs.shape)
        return outputs + inputs[:,:,-outputs.shape[2]:], outputs

class WaveNet(nn.Module):
    def __init__(self, num_classes, channels_in, channels_out=128, num_layers=3, dilations=[1,2,4]): # dilations=[1,2,4]
        super(WaveNet, self).__init__()
        self.num_layers = num_layers
        self.conv1d = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels_out)

        self.resnet_block = nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.conv1d_out = nn.Conv1d(channels_out, channels_out, kernel_size=1)
        self.get_logits = nn.Conv1d(in_channels=channels_out, out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self.bn(self.conv1d(inputs))
        x = torch.tanh(x)
        out = 0.0
        for _ in range(self.num_layers):
            for layer in self.resnet_block:
                x, out = layer(x)
                out += out
        out = torch.tanh(self.bn(self.conv1d_out(out)))

        logits = self.get_logits(out)

        return logits

if __name__ == '__main__':
    model = WaveNet(num_classes=8, channels_in=16)
    model.eval()
    input = torch.rand([4,16,128]) # [4,16,128] may be too short. maybe there is some error in padding.
    print(model(input))



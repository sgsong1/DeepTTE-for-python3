import torch
import torch.nn as nn
import torch.nn.functional as F

from . import GeoConv  # 사용자 정의 모듈로 가정
import numpy as np

class Net(nn.Module):
    '''
    attr_size: the dimension of attr_net output
    pooling options: last, mean, attention
    '''
    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method='attention', rnn='lstm'):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method

        self.geo_conv = GeoConv.Net(kernel_size=kernel_size, num_filter=num_filter)

        # num_filter: output size of each GeoConv + 1: distance of local path + attr_size: output size of attr component
        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=num_filter + 1 + attr_size, 
                               hidden_size=128, 
                               num_layers=2, 
                               batch_first=True)
        elif rnn == 'rnn':
            self.rnn = nn.RNN(input_size=num_filter + 1 + attr_size, 
                              hidden_size=128, 
                              num_layers=1, 
                              batch_first=True)

        if pooling_method == 'attention':
            self.attr2atten = nn.Linear(attr_size, 128)

    def out_size(self):
        # Return the output size of the spatio-temporal component
        return 128

    def mean_pooling(self, hiddens, lens):
        # Pad_packed_sequence outputs padded hidden states (zeros). We can simply sum and normalize by lengths.
        hiddens = torch.sum(hiddens, dim=1, keepdim=False)

        lens = torch.FloatTensor(lens).to(hiddens.device)

        lens = torch.unsqueeze(lens, dim=1)
        hiddens = hiddens / lens

        return hiddens

    def attent_pooling(self, hiddens, lens, attr_t):
        attent = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)

        # hidden: b*s*f, atten: b*f*1, alpha: b*s*1 (s is length of sequence)
        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(-alpha)

        # No need for masking since PyTorch pads hidden states with zeros.
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)

        return hiddens

    def forward(self, traj, attr_t, config):
        # Apply GeoConv
        conv_locs = self.geo_conv(traj, config)

        attr_t = torch.unsqueeze(attr_t, dim=1)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1], ))

        # Concatenate the loc_conv and the attributes
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)

        lens = list(map(lambda x: x - self.kernel_size + 1, traj['lens']))

        # Pack the padded sequence
        packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first=True, enforce_sorted=False)

        # RNN forward pass
        packed_hiddens, (h_n, c_n) = self.rnn(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)

        elif self.pooling_method == 'attention':
            return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)

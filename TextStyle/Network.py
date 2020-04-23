import torch
import torch.nn as nn

class TextStyleNet(nn.Module):
    def __init__(self, num_chars, num_layers, num_nodes=512, dropout=0.1):
        super(TextStyleNet, self).__init__()

        input_shape = num_chars

        self.lstm = nn.LSTM(num_chars, num_nodes, num_layers,dropout=dropout)
       
    def forward(self, x):
        raise NotImplementedError

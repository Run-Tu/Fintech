import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
        LSTM Net可视化https://zhuanlan.zhihu.com/p/139617364(可反复观看)
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, directions=1):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.directions = directions # 单向lstm

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, drop=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    
    def init_hidden_states(self, batch_size, device):
        """
            LSTM的h层和c层的shape
            h和c应该是一个shape。2层每层是(batch_size * hidden_size)
            双向的话每一层有两个方向,所以是2层每层是(2, batch_size * hidden_size),一共就是(4, batch_size * hidden_size)
        """
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)

        return (torch.zeros(state_dim).to(device), torch.zeros(state_dim).to(device))

    
    def forward(self, x, states):
        x, (h,c) = self.lstm(x, states)
        out = self.linear(x)

        return out, (h,c)
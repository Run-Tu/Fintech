import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
        LSTM Net可视化https://zhuanlan.zhihu.com/p/139617364
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, directions=1):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.directions = directions

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    
    def init_hidden_states(self, batch_size, device):
        """
            初始化LSTM的h层和c层的参数
            h和c是一个shape,(num_layers*direction, batch_size, hidden_size)
        """
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)

        return (torch.zeros(state_dim).to(device), torch.zeros(state_dim).to(device))

    
    def forward(self, x, states):
        x, (h,c) = self.lstm(x, states)
        out = self.linear(x)

        return out, (h,c)
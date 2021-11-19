import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_size, n_hidden):
        """
        Initializes multi-layer perceptron object.    
        Args:
            input_size: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        Function:
            add_module(name, linear): 添加网络每一层，并且可以给每一层增加单独的名字
        """
        super().__init__()
        self.num_hid = len(n_hidden)
        self.layers = list()
        for i in range(self.num_hid):
            name = "hidden_layer" + str(i)
            linear = None
            if i == 0:
                linear = nn.Linear(input_size, n_hidden[i])
            else:
                linear = nn.Linear(n_hidden[i-1], n_hidden[i])
            self.add_module(name, linear)
            self.layers.append(linear)
        self.output_layer = nn.Linear(n_hidden[-1], 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
    

    def forward(self, input):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Function：
            torch.clamp(input, min, max, out=None) -> Tensor: 将每个tensor夹紧到区间
            [min, max] 
        Returns:
            out: output of the network
        """
        linear_input = input
        for layer in self.layers:

            # linear_input = layer(linear_input).clamp(min=0) 考虑用-1填充Nan，让模型捕捉空值信息，所以不用clamp
            linear_input = torch.sigmoid(layer(linear_input))
        out_put = self.output_layer(linear_input)
        # logits = torch.sigmoid(out_put)
        
        return out_put
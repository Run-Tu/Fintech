import torch
import torch.nn as nn
import math
from preprocessing.preprocessor import Data_Process
# entroy=nn.CrossEntropyLoss()
# input=torch.Tensor([[-0.7715, -0.6205,-0.2562]])
# target = torch.tensor([0])
# output = entroy(input, target)
# print("test input size is ", input.size())
# print("test target size is ", target.size())
# print(output)
# #根据公式计算

# a = [0.2, 0.3, 0.5, 0.6, 0.7]
# b = list(map(lambda x: 1 if x>=0.5 else 0, a))
# print(b)
train_process_tensor, y_train = Data_Process('train')
print("train data kind of 1 ", sum(y_train))
print("whole train data ", len(y_train))
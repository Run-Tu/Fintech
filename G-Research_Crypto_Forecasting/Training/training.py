import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import io
import json
import time
import requests
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None


class Trainner():
    """
        封装训练方法
    """
    def __init__(self):
        """
            初始化参数：
        """
        pass


    def save_checkpoint(self, epoch, min_val_loss, model_state, opt_state):
        """
            通过建立字典的方式保存参数,调用torch.save(state, dir)
            pytorch实现断点训练参考：https://zhuanlan.zhihu.com/p/133250753
        """
        print(f"New minimum reached at epoch #{epoch+1}, saving model state...")
        checkpoint = {
            'epoch': epoch+1,
            'min_val_loss': min_val_loss,
            'model_state': model_state,
            'opt_state': opt_state
        }

        torch.save(checkpoint, "../checkpoint/model_state.pt")
    

    def load_checkpoint(self,path, model, optimizer):
        # load check point
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_val_loss']
        model.load_state_dict(checkpoint['opt_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])

        return model, optimizer, epoch, min_val_loss


    def training(self,model, batch_size, epochs, training_dl, validation_dl,criterion, optimizer,validate_every=2):
        """
            batch_size参数传递问题？
            参数：
            validate_every：每x个epoch验证一次loss
        """
        # 是否可以将device设置为全局变量？简化每次调用
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        training_losses = []
        validation_losses = []
        min_validation_loss = np.inf

        # set to train mode
        model.train()

        for epoch in tqdm(range(epochs)):
            # Initialize hidden and cell states with dimension:
            # (num_layers * num_directions, batch, hidden_size)
            states = model.init_hidden_states(batch_size)
            running_training_loss = 0.0
        
            # Training
            for idx, (x_batch, y_batch) in enumerate(training_dl):
                    # Convert to Tensors
                    x_batch = x_batch.float().to(device)
                    y_batch = y_batch.float().to(device)

                    # Truncated Backpropagation
                    states = [state.detach() for state in states]

                    optimizer.zero.grad()
                    
                    # Make prediction
                    output, states = model(x_batch, states)

                    # Calculate loss
                    loss = criterion(output[:, -1, :], y_batch)
                    loss.backward()
                    running_training_loss += loss
            
            # Average loss across timesteps
            training_losses.append(running_training_loss / len(training_dl))

            if epoch % validate_every == 0:
                # Set to eval mode
                model.eval()
                # 这里为什么在验证的时候做了一个参数初始化？
                validation_states = model.init_hidden_states(batch_size)
                running_validation_loss = 0.0

                for idx, (x_batch, y_batch) in enumerate(validation_dl):
                        # Convert to Tensors
                        x_batch = x_batch.float().to(device)
                        y_batch = y_batch.float().to(device)
                        # 今天先写到这2022/1/5
                        validation_states = [state.detach() for state in validation_states]





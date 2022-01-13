import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# 忽略处理数据帧切片时的chained_assignment警告(参考：https://stackoverflow.com/questions/37841525/correct-way-to-set-value-on-a-slice-in-pandas)
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

        torch.save(checkpoint, "./models/checkpoint/model_state.pt")
    

    def load_checkpoint(self,path, model, optimizer):
        # load check point
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_val_loss']
        model.load_state_dict(checkpoint['opt_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])

        return model, optimizer, epoch, min_val_loss


    def training(self, model, batch_size, device, epochs, training_dl, validation_dl, criterion, optimizer, validate_every=2):
        """
            batch_size参数传递问题？
            参数：
            validate_every：每x个epoch验证一次loss
            可以尝试通过字典传参**params
        """
        training_losses = []
        validation_losses = []
        min_validation_loss = np.inf

        # set to train mode
        model.train()

        def plotting_loss(training_losses=None, validation_losses=None):
            """
                plotting train | validtation loss
            """
            if training_losses:
                epoch_count = range(1, len(training_losses)+1)
                plt.plot(epoch_count, training_losses, 'r--')
                plt.title(['Training Loss'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.show()
            if validation_losses:
                epoch_count = range(1, len(validation_losses)+1)
                plt.plot(epoch_count, validation_losses, 'b--')
                plt.title(['Validation Loss'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.show()


        for epoch in tqdm(range(epochs)):
            # Initialize hidden and cell states with dimension:
            # (num_layers * num_directions, batch, hidden_size)
            states = model.init_hidden_states(batch_size, device)
            print("states size is", states[0].size())
            running_training_loss = 0.0
        
            # Training
            for _ , (x_batch, y_batch) in enumerate(training_dl):
                    # Convert to Tensors
                    x_batch = x_batch.float().to(device)
                    y_batch = y_batch.float().to(device)

                    # Truncated Backpropagation
                    """
                        detach()用于阻断反向传播,为什么要阻断反向传播？
                    """
                    # states = [state.detach() for state in states]
                    states = [state for state in states] # 不阻断试一下

                    optimizer.zero_grad()
                    
                    # Make prediction
                    # 训练时要将drop_last设置为True,否则最后一个batch样本数量不够,维度对不上
                    # 参考:https://blog.csdn.net/weixin_43935696/article/details/118970831
                    output, states = model(x_batch, states)

                    # Calculate loss
                    loss = criterion(output[:, -1, :], y_batch)
                    # 模型有两个输出,backward时需要设置retain_graph参数
                    # 参考:https://blog.csdn.net/weixin_44058333/article/details/99701876
                    loss.backward(retain_graph=True)
                    running_training_loss += loss.item()
            
            # Average loss across timesteps
            training_losses.append(running_training_loss / len(training_dl))

            if epoch % validate_every == 0:
                # Set to eval mode
                model.eval()
                # 这里为什么在验证的时候做了一个参数初始化？
                validation_states = states # 沿用训练时的states参数
                # validation_states = model.init_hidden_states(batch_size, device)
                running_validation_loss = 0.0

                for _ , (x_batch, y_batch) in enumerate(validation_dl):
                        # Convert to Tensors
                        x_batch = x_batch.float().to(device)
                        y_batch = y_batch.float().to(device)

                        # Truncated Backpropagation
                        validation_states = [state.detach() for state in validation_states] # 验证集不让反向传播
                        # validation_states = [state for state in validation_states] # 保持梯度反向传播
                        output, validation_states = model(x_batch, validation_states)
                        # DEBUG 记得查看output的shape
                        validation_loss = criterion(output[:, -1, :], y_batch)
                        running_validation_loss += validation_loss.item()
                
            validation_losses.append(running_validation_loss / len(validation_dl))
            # Reset to training model
            model.train()

            is_best = running_validation_loss / len(validation_dl) < min_validation_loss

            if is_best:
                min_validation_loss = running_validation_loss / len(validation_dl)
                self.save_checkpoint(
                                     epoch+1, 
                                     min_validation_loss, 
                                     model.state_dict(),
                                     optimizer.state_dict()  
                                    )  
            
            # Visualize loss
            plotting_loss(training_losses)
            plotting_loss(validation_losses)
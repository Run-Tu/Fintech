import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data.dataloader import DataLoader
from data_process.processor import get_train_and_valid_df
from Training.trainner import Trainner
from models.Net.lstm import LSTM
from DataSet.CryptoDataset import CryptoDataset

# DEVICE
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

############
EPOCHS        = 10
DROPOUT       = 0.2
DIRECTIONS    = 1
NUM_LAYERS    = 2
BATCH_SIZE    = 6
OUTPUT_SIZE   = 1
SEQ_LENGTH    = 60 # 时间戳单位是s，捕捉60s的信息
NUM_FEATURES  = 12 # 特征数量可适当加特征
HIDDEN_SIZE   = 64
LEARNING_RATE = 0.0001
# h和c的shape(num_layers, batch_size, hidden_size)
STATE_DIM     = NUM_LAYERS * DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE
TARGET        = "target"
FEATURES      = ["Asset_ID","Count","Open","High","Low","Close",
                 "Volume","VWAP","up_shadow","low_shadow",
                 "five_min_log_return","abs_one_min_log_return"]
############
DATALOADER_PARAMS = {
                'batch_size':6,
                'shuffle':False,
                'drop_last':False,
                'num_workers':2
              }
params = {
            'epoch':10 ,
            'dropout':0.2 ,
            'directions':1 ,
            'num_layers':2 ,
            'batch_sie':6
            }


if __name__ == '__main__':
    # step1:获取处理好的数据集
    """
        step1可封装，返回training_dl 和 validation_dl
    """
    train_df, valid_df = get_train_and_valid_df('./data/train.csv', rows=1800)
    training_ds = CryptoDataset(train_df, seq_length=6, features=FEATURES, target="target")
    training_dl = DataLoader(training_ds, **DATALOADER_PARAMS)

    validation_ds = CryptoDataset(valid_df, seq_length=6, features=FEATURES, target='target')
    validation_dl = DataLoader(validation_ds, **DATALOADER_PARAMS)
    # step2:利用封装好的Trainner类训练数据
    """
        参数可参考ccks的方式进行整理
    """
    trainer = Trainner()
    LSTM_model = LSTM(
                    input_size = NUM_FEATURES, # LSTM input_size即num_features特征维度
                    hidden_size = HIDDEN_SIZE,
                    num_layers = NUM_LAYERS,
                    output_size = OUTPUT_SIZE,
                    dropout_prob = DROPOUT
                    ).to(DEVICE)
    trainer.training(
                      model = LSTM_model, 
                      batch_size = BATCH_SIZE,
                      device = DEVICE,
                      epochs = EPOCHS,
                      training_dl = training_dl,
                      validation_dl = validation_dl,
                      criterion = nn.MSELoss(),
                      # AdamW拿模型参数的时候为什么model.linear.parameters()?
                      optimizer = optim.AdamW(LSTM_model.linear.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
                      )
    # step3:加载训练好的模型并预测
    """
        load_checkpoint传入的model如果修改参数会报什么错？
    """
    path = './models/checkpoint/model_state.pt'
    model, optimizer, start_epoch, valid_loss_min = trainer.load_checkpoint(path, LSTM_model, optimizer)
    print("model = ", model)
    print("optimizer = ", optimizer)
    print("start_epoch = ", start_epoch)
    print("valid_loss_min = ", valid_loss_min)
    print("valid_loss_min = {:.6f}".format(valid_loss_min))

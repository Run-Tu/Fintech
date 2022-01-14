"""
    1、调研early-stopping
    2、缺少predict预测模块
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from data_process.processor import get_train_and_valid_df
from Training.trainner import Trainner
from models.Net.lstm import LSTM
from DataSet.CryptoDataset import CryptoDataset

# DEVICE
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

def get_train_and_valid_dl(args):
        """
            Return Train and Validation Dataloader object
        """
        train_df, valid_df = get_train_and_valid_df('./data/train.csv', rows=1800)
        features = ["Asset_ID","Count","Open","High","Low","Close",
                     "Volume","VWAP","up_shadow","low_shadow",
                     "five_min_log_return","abs_one_min_log_return"]
        dataloader_params = {
                              'batch_size' : args.batch_size,
                              'shuffle' : False,
                              'drop_last' : True,
                              'num_workers' : 2 
                            }
        training_ds = CryptoDataset(train_df, 
                                    seq_length=args.seq_length, 
                                    features=features, 
                                    target=args.target)
        training_dl = DataLoader(training_ds, **dataloader_params)

        validation_ds = CryptoDataset(valid_df, 
                                      seq_length=args.seq_length, 
                                      features=features, 
                                      target=args.target)
        validation_dl = DataLoader(validation_ds, **dataloader_params)
        
        return training_dl, validation_dl


def main(args):
    training_dl, validation_dl = get_train_and_valid_dl(args)
    LSTM_model = LSTM(
                    input_size = args.input_size, # LSTM input_size即num_features特征维度
                    hidden_size = args.hidden_size,
                    num_layers = args.num_layers,
                    output_size = args.output_size,
                    dropout_prob = args.dropout
                    ).to(DEVICE)
    trainer = Trainner()
    optimizer = optim.AdamW(LSTM_model.linear.parameters(), lr=args.learning_rate, weight_decay=0.01)
    trainer.training(
                      model = LSTM_model, 
                      batch_size = args.batch_size,
                      device = DEVICE,
                      epochs = args.epoch,
                      training_dl = training_dl,
                      validation_dl = validation_dl,
                      criterion = nn.MSELoss(),
                      # AdamW拿模型参数的时候为什么model.linear.parameters()?
                      optimizer = optimizer
                      )

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--seq_length', type=int, default=60)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of LSTM hidden layers.')
    parser.add_argument('--input_size', type=int, default=12,
                        help='number of features in the data.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM hidden layer size')
    parser.add_argument('--output_size', type=int, default=1,
                        help='LSTM hidden layer output size')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--directions', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=False,
                        help='If shuffle torch DataLoader batch_size')
    parser.add_argument('--drop_last', type=bool, default=True,
                        help='Drop the last incomplete batch')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Num_worker to load torch Dataloader')
    parser.add_argument('--target', type=str, default='target',
                        help='The target used to create the dataset')
    args = parser.parse_args()
    main(args)
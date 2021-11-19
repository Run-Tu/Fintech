import pandas as pd
import torch


def Data_Process(data_type):
    """   
    Data_Process Function
    - target:change csv data to tensor data(train_process、testA_process)
    """
    assert data_type in ['train','dev','test']
    data_path = {'train_process_path':'/mnt/Loan_default_prediction/dataset/train_process.csv',
                 'testA_process_path':'/mnt/Loan_default_prediction/dataset/testA_process.csv'}

    if data_type in ['train', 'dev']:
        data_process = pd.read_csv(data_path['train_process_path'])
        if data_type == 'train':
            train_process_tensor = torch.tensor(data_process.iloc[:600000,:-1].values).float()
            y_train= torch.tensor(data_process.iloc[:600000,-1].values).float()

            return train_process_tensor, y_train
        else:
            dev_process_tensor = torch.tensor(data_process.iloc[600000:,:-1].values).float()
            y_dev = torch.tensor(data_process.iloc[600000:,-1].values).float()

            return dev_process_tensor,y_dev

    if data_type == 'test':
        testA_process = pd.read_csv(data_path['testA_process_path'])
        testA_process_tensor = torch.tensor(testA_process.iloc[:,:].values).float()

        return  testA_process_tensor
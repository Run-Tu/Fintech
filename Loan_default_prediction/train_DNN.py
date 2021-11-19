import argparse
import numpy as np
import torch
import torch.utils.data as Data
from net.DNN import DNN
from preprocessing.preprocessor import Data_Process
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# Parameter Group
parameter_v1 = {
                'DNN_HIDDEN_UNITS_DEFAULT' : '512,256,128',
                'LEARNING_RATE_DEFAULT' : 1e-2,
                'MAX_EPOCHS_DEFAULT' : 10,
                'BATCH_SIZE' : 4096
                }
parameter_v2 = {
                'DNN_HIDDEN_UNITS_DEFAULT' : '512,256,128',
                'LEARNING_RATE_DEFAULT' : 1e-2,
                'MAX_EPOCHS_DEFAULT' : 2,
                'BATCH_SIZE' : 1024
                }
parameter_v3 = {
                'DNN_HIDDEN_UNITS_DEFAULT' : '256,128,64',
                'LEARNING_RATE_DEFAULT' : 1e-2,
                'MAX_EPOCHS_DEFAULT' : 2,
                'BATCH_SIZE' : 512
                }


def auc(dnn, x, y):
    """
        caculate auc socre
    """
    y_pred = torch.squeeze(dnn.forward(x).detach()).numpy()
    auc = roc_auc_score(y_true=y, y_score=y_pred)

    return auc


def train(dnn, x, y, args):
    """
    Performs training and evaluation of DNN model.
    - step1: wrap train_data to loader 
    - step2: train DNN model by shuffle
    """
    # step1: wrap train_data to loader
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = args.batch_size,
        shuffle = True,
    )
    losses = []

    # step2: train DNN model by shuffle
    for epoch in range(args.max_epochs):   
        for _, (batch_x, batch_y) in enumerate(loader):
            batch_y_pred = torch.squeeze(dnn.forward(batch_x))
            # print("batch_y_pred is ", batch_y_pred)
            # print("batch_x is {} , batch_y is {}".format(batch_x,batch_y))
            loss = dnn.criterion(batch_y_pred, batch_y)
            dnn.optimizer.zero_grad()
            loss.backward()
            dnn.optimizer.step()
            losses.append(loss.cpu().data.item())
        print ('Epoch : %d/%d,  Loss: %.4f'%(epoch+1, args.max_epochs, np.mean(losses)))

    return dnn


def main(args):
    """
    Main function
    - step1: Get train & test data
    - step2: Feeding data to train DNN Net
    - step3: Evaluation on dev data
    - step4: Predict on testA data
    """
    # step1: get train_data and dev_data
    train_process_tensor, y_train = Data_Process('train')
    dev_process_tensor, y_dev = Data_Process('dev')
    testA_process_tensor = Data_Process('test')

    # step2: Feeding data to train DNN Net
    hiddens_temp = args.dnn_hidden_units.split(",")
    hiddens = list()
    for value in hiddens_temp:
        hiddens.append(int(value))
    dnn = DNN(48, hiddens)
    dnn.optimizer = torch.optim.SGD(dnn.parameters(), lr=args.learning_rate)
    print("create finish")
    dnn_trained = train(dnn, train_process_tensor, y_train, args)

    # step3: evaluation on dev data
    auc_trained_result = auc(dnn_trained, dev_process_tensor, y_dev)
    print("Trained DNN auc_result is %f"%(auc_trained_result))

    auc_result = auc(dnn, dev_process_tensor, y_dev)
    print("NonTrained DNN auc_result is %f"%(auc_result))
    # step4: predict on testA data
    result = torch.squeeze(dnn_trained.forward(testA_process_tensor).detach()).numpy()

    return result

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = parameter_v1['DNN_HIDDEN_UNITS_DEFAULT'],
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = parameter_v1['LEARNING_RATE_DEFAULT'],
                      help='Learning rate')
    parser.add_argument('--max_epochs', type = int, default = parameter_v1['MAX_EPOCHS_DEFAULT'],
                      help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type = int, default = parameter_v1['BATCH_SIZE'],
                      help='Number of epochs to run trainer.')
    args = parser.parse_args()
    main(args)

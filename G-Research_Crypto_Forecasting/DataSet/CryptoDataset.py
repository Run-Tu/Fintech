from torch.utils.data import Dataset


class CryptoDataset(Dataset):
    """
        Time Series dataset
    """

    def __init__(self, csv_file, seq_length, features, target):
        self.csv_file = csv_file
        self.seq_length = seq_length
        self.features = features
        self.data_length = len(csv_file)
        self.target = target

        self.metrics = self.create_xy_pairs()


    def create_xy_pairs(self):
        """
            滑动窗口得到训练数据集
        """
        pairs = []
        for idx in range(self.data_length - self.seq_length):
            x = self.csv_file[idx : idx+self.seq_length][self.features].values
            y = self.csv_file[idx+self.seq_length : idx+self.seq_length+1][self.target].values
            pairs.append((x,y))
        
        return pairs


    def __len__(self):
        """
            torch.utils.data.Dataset是抽象类
            实现抽象类需要重写__len__方法
        """

        return len(self.metrics)


    def __getitem__(self, idx):
        """
            torch.utils.data.Dataset是抽象类
            实现抽象类需要重写__getitem__方法
        """

        return self.metrics[idx]

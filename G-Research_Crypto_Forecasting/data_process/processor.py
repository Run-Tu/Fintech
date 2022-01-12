import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
FEATURES_COL = ["Asset_ID","Count","Open","High","Low","Close","Volume","VWAP","datatime"]

def get_train_and_valid_df(path, rows):
    """
        集成load_data,create_features,prepare_split三个功能
        返回处理好的train_df,valid_df
        Args:
            path:数据路径
            rows:读取数据的行数
    """
    def load_data(path, rows=1800):
        """
            加载数据
            参数：
                path:数据路径
                rows:读取数据时选取多少行
        """
        df_train = pd.read_csv(path, nrows=rows)
        df_train.dropna(axis=0, inplace=True)
        df_train.sort_values("timestamp", inplace=True) # 将时间戳排序，需要按时间划分时间序列数据集
        df_train['datatime'] = pd.to_datetime(df_train['timestamp'], units='s') # 转换时间戳，单位s

        return df_train


    def create_features(df, label=False):
        """
            特征工程
            参数:
            label:是否处理标签值
        """
        def upper_shadow(df):
            """
                upper_shadow:当天最高值 - 开盘和收盘价的最大值
            """

            return df['High'] - np.maxinum(df['close'],df['open'])
        

        def lower_shadow(df):
            """
                lower_shadow:开盘和收盘价格的最低值 - 当天最低值
            """

            return np.minimum(df['close'],df['open']) - df['low']
        

        def log_return(series, periods=1):

            return np.log(series).diff(periods)


        def fill_nan_inf(df):

            df = df.fillna(0) # fillna
            df = df.replace([np.inf, -np.inf], 0) # fillinf

            return df
            

        # Build Features
        up_shadow = upper_shadow(df)
        low_shadow = lower_shadow(df)
        five_min_log_return = log_return(df.VWAP, periods=5)
        abs_one_min_log_return = log_return(df.VWAP, periods=1).abs()
        features = df[FEATURES_COL]

        # concat
        X = pd.concat([features, up_shadow, low_shadow, five_min_log_return, abs_one_min_log_return], axis=1)
        X.columns = FEATURES_COL + ['up_shadow', 'low_shadow', 'five_min_log_return', 'abs_one_min_log_return']
        X = fill_nan_inf(X)

        if label:
            y = df['Target']
            y = fill_nan_inf(y)
            X['target'] = y
        
        def reduce_mem(df):
            """
                reduce machine memory
            """
            for col in df.columns:
                if col != 'datetime':
                    df[col] = df[col].astype('float32')

            return df

        return reduce_mem(X)


    def prepare_split(df):

        train_df = df[df['datetime'] <= '2018-06-03 00:00:00']
        valid_df = df[df['datetime']  > '2018-06-03 00:00:00']

        print(f"Training data size: {train_df.shape}",
            f"Validation data size: {valid_df.shape}")

        return train_df, valid_df
    

    df_train = load_data(path, rows)
    df_train = create_features(df_train, label=True)
    train_df, valid_df = prepare_split(df_train)

    return train_df, valid_df
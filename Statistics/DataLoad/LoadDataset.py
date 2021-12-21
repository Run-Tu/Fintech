from lib.features_utils import add_features

def get_slide_window_data(df,windows_size:list=[1,3]):
    """
        Tips:
            数据下沉，以历史月份数据作为特征，当前月份目标值作为标签来预测
            单值下沉用shift,聚合下沉用rolling
        Get train&valid data by slide window
        Arguments:
            df: The target dataframe to be divided
            window_size(list): slide window size 1:month 3:quarterly
    """
    df = add_features(df)
    target = df['term_loan_scale']
    for window_size in windows_size:
        df['loan_size_lag_'+str(window_size)] = df['loan_size'].shift(window_size)
        df['term_loan_size_lag_'+str(window_size)] = df['term_loan_size'].shift(window_size)
        df['term_loan_scale_lag_'+str(window_size)] = df['term_loan_scale'].shift(window_size)
        df['com_loan_size_lag_'+str(window_size)] = df['com_loan_size'].shift(window_size)
        df['term_com_loan_size_lag_'+str(window_size)] = df['term_com_loan_size'].shift(window_size)
        df['term_com_loan_scale_lag_'+str(window_size)] = df['term_com_loan_scale'].shift(window_size)
        df['loan_difference_lag_'+str(window_size)] = df['loan_difference'].shift(window_size)
        df['com_loan_difference_lag_'+str(window_size)] = df['com_loan_difference'].shift(window_size)
        if window_size !=1 :
            df['loan_size_lag_'+str(window_size)+'_mean'] = df['loan_size'].rolling(window_size).mean()
            df['term_loan_size_lag_'+str(window_size)+'_mean'] = df['term_loan_size'].rolling(window_size).mean()
            df['term_loan_scale_lag_'+str(window_size)+'_mean'] = df['term_loan_scale'].rolling(window_size).mean()
            df['com_loan_size_lag_'+str(window_size)+'_mean'] = df['com_loan_size'].rolling(window_size).mean()
            df['term_com_loan_size_lag_'+str(window_size)+'_mean'] = df['term_com_loan_size'].rolling(window_size).mean()
            df['term_com_loan_scale_lag_'+str(window_size)+'_mean'] = df['term_com_loan_scale'].rolling(window_size).mean()
            df['loan_difference_lag_'+str(window_size)+'_mean'] = df['loan_difference'].rolling(window_size).mean()
            df['com_loan_difference_lag_'+str(window_size)+'_mean'] = df['com_loan_difference'].rolling(window_size).mean()
    
    df.drop(['date','loan_difference','com_loan_difference',
             'loan_size','term_loan_size',
             'term_loan_scale','com_loan_size',
             'term_com_loan_size','term_com_loan_scale'],
             axis=1,inplace=True)

    X_train = df.iloc[max(windows_size)+1:-1,:]
    y_train = target[max(windows_size)+1:-1]
    X_valid = df.iloc[-1:,:]
    y_valid = target[-1:]

    
    return  X_train,y_train,X_valid,y_valid
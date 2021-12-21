from lightgbm import LGBMRegressor
from lib.features_utils import *
from DataSet.LoadDataset import get_slide_window_data

if __name__ == '__main__':
    df = excel_to_df('17-21中长期贷款汇总.xlsx')
    X_train,y_train,X_valid,y_valid = get_slide_window_data(df)

    model = LGBMRegressor(n_estimators=10)
    model.fit(X_train,y_train)
    print("The Result is",model.predict(X_valid))

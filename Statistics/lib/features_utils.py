import pandas as pd
from openpyxl_utils import get_sheet_object

def excel_to_df(excel_path,save_path=None):
    """
        Arguments:
            excel_path: The path of the excel file
            save_path: The path which saved the converted dataframe 
    """
    ori_sheet = get_sheet_object(excel_path)
    dataframe_value = []
    for row in range(5,ori_sheet.max_row+1):
        row_list = []
        for col in range(1,ori_sheet.max_column+1):
            row_list.append(ori_sheet.cell(row,col).value)
        dataframe_value.append(row_list)

    df = pd.DataFrame(dataframe_value,
                      columns=['date','loan_size','term_loan_size','term_loan_scale',
                               'com_loan_size','term_com_loan_size','term_com_loan_scale'])
    if save_path:
        df.to_csv(save_path)

    return df


def add_features(df):
    """
        Feature Enginering
        Arguments:
            df: The converted dataframe
    """
    df['loan_difference'] = df['loan_size'] - df['term_loan_size']
    df['com_loan_difference'] = df['com_loan_size'] - df['term_com_loan_size']
    df['year'] = df['date'].apply(lambda x: x[:4]).astype('int32')
    df['month'] = df['date'].apply(lambda x: x[6] if x[5]=='0' else x[5:7]).astype('int32')
    df['day'] = df['date'].apply(lambda x: x[-3:-1]).astype('int32')
    df['First_half_year'] = df['year'].apply(lambda x: 1 if x in [1,2,3,4,5,6] else 0)
    df['Second_half_year'] = df['year'].apply(lambda x: 1 if x in [7,8,9,10,11,12] else 0)
    quarterly_map = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
    df['Quarterly'] = df['month'].map(quarterly_map)

    return df





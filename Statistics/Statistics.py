from lib.openpyxl_utils import (
    get_target_list,
    save_modify_sheet
)
import os
"""
    openpyxl教程：https://blog.csdn.net/David_Dai_1108/article/details/78702032
    Sheet操作：
        1.加载excel文件(默认可读写);通过参数data_only只显示表格结果数字,过滤公式
        wb = load_workbook(data_list[0], data_only=True)
        2.通过get_sheet_names()获得当前xlsx文件的所有sheet
        name_list = wb.get_sheet_names()
        3.根据sheet名字获得sheet内容
        my_sheet = wb.get_sheet_by_name('Sheet1')
    Cell操作：
        4.获取某个单元格的值,先字母再数字,先列再行
        target = my_sheet['B30']
"""
# Get All File Name in Specified Path
data_path_list = ['../xlsx_data/'+data_path for data_path in os.listdir('../xlsx_data')]
# Find Target Value In Sheet           
Date, Total_Load_Size, Mid_To_Long_Term_Total_Load_Size,\
Corporate_Loan_Size, Mid_To_Long_Term_Corporate_Loan_Size,\
Mid_Loan_Scale, Mid_Corporate_Loan_Scale = get_target_list(data_path_list)
row_index_list = [row for row in range(5,len(Total_Load_Size)+5)]
row_index_and_Required_Data = list(zip(row_index_list,Date,Total_Load_Size,Mid_To_Long_Term_Total_Load_Size,
                                       Mid_Loan_Scale,Corporate_Loan_Size,
                                       Mid_To_Long_Term_Corporate_Loan_Size,Mid_Corporate_Loan_Scale))


if __name__ == '__main__':
    # Save Result
    save_modify_sheet(final_sheet_path='../17-21中长期贷款汇总.xlsx',
                    data_path='../17-21中长期贷款汇总.xlsx',
                    required_data=row_index_and_Required_Data
                    )

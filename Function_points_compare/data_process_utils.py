import os
import json
import datetime
import logging
import pandas as pd
from collections import defaultdict

# logging
TODAY = datetime.date.today()
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y/%m/%d %H:%M:%S %P"
if not os.path.exists("./logs"):
    os.mkdirs("./logs")
logging.basicConfig(filename=f"./logs/{TODAY}.log", level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def read_work_book(path:str):
    """read excel file
    
    Args:
        path (_str_): excel file path

    Returns:
        workbook: from pd.read_excel()
    """
    workbook = []
    sheet_names = pd.ExcelFile(path).sheet_names
    try:
        if "规模估算-技术确认" in sheet_names:
            workbook = pd.read_excel(path, sheet_name="规模估算-技术确认", engine="openpyxl")
        if "规模估算" in sheet_names:
            workbook = pd.read_excel(path, sheet_name="规模估算", engine="openpyxl")
    except ValueError:
        logging.info(f"excel读取异常,sheet页中没有规模估算-技术确认或规模估算,错误数据路径{path}")
    except Exception as e:
        logging.info(f"错误信息是{e},错误数据路径{path}")
    
    return workbook


def process_excel_data(workbook):
    """
    Args:
        workbook : from pd.read_excel()

    Returns:
        system : system name
        system_info : spliced system function points by 'SEP' 
    """
    if workbook is None:

        return "Workbook is null Please check read_work_book()"
    
    workbook_columns = ["系统名称" if col=="所属系统" else col for col in workbook.iloc[3,:].values]
    workbook.columns = workbook_columns
    subset_with_systemName = ["系统名称","二级模块","三级模块","功能点计数项名称","类别"]
    subset_without_systemName = ["子需求单号","二级模块","三级模块","功能点计数项名称","类别"]

    try: 
        workbook["系统名称"].fillna(method="ffill", inplace=True)
        workbook.dropna(inplace=True, subset=subset_with_systemName)
    except KeyError:
        workbook.dropna(inplace=True, subset=subset_without_systemName)
    workbook = workbook.iloc[1:,:].reset_index(drop=True)

    system = system_info = []
    if "系统名称" in workbook_columns:
        for _, row in workbook.iterrows():
            system.append(row["系统名称"])
            info = "[SEP]".join([row["二级模块"], 
                                 row["三级模块"], 
                                 row["功能点计数项名称"],
                                 row["类别"]])
            
            system_info.append(info)

    else:
        for _, row in workbook.iterrows():
            system.append("default")
            info = "[SEP]".join([row["子需求单号"],
                                 row["二级模块"],
                                 row["三级模块"],
                                 row["功能点计数项名称"],
                                 row["类别"]])
            system_info.append(info)
            
    return system, system_info


def get_systemFunctionInfo_map(data_path:list):
    """Read All Excel Data

    Args:
        data_path (_list_): The folder where is stored excel files

    Returns:
        _dict : {system:[all system_info]}
    """
    _dict = defaultdict(list)

    for data in data_path:
        workbook = read_work_book(os.path.join("./original_data/", data))
        all_system, all_system_info = process_excel_data(workbook)
        logging.info(f"all_system is {all_system} all_system_info is {all_system_info}")
        for system, system_info in zip(all_system, all_system_info):
            logging.info(f"system is {system} system_info is {system_info}")
            _dict[system].append(system_info)

    return _dict


def main():
    data_path= os.listdir("./original_data/")
    systemFunctionInfo_map = get_systemFunctionInfo_map(data_path)

    with open("./db_data/db.json", 'w') as json_file:
        json.dump(systemFunctionInfo_map, json_file, ensure_ascii=False)


if __name__ == "__main__":
    main()
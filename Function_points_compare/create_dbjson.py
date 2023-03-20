"""
    put original xlsx data to the original_data folder, use create_dbjson.py generate {'system':[all_system_info]} db.json
    command: python create_dbjson.py
"""
import os
import json
from data_utils import get_systemFunctionInfo_map


def main():
    data_path_lst = os.listdir("./original_data/")
    systemFunctionInfo_map = get_systemFunctionInfo_map(data_path_lst, path="./original_data/")

    with open("./db_data/db.json", 'w') as json_file:
        json.dump(systemFunctionInfo_map, json_file, ensure_ascii=False)


if __name__ == "__main__":
    main()
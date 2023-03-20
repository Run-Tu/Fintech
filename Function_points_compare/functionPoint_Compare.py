"""
    self compare:
    python functionPoint_Compare.py --self_compare -ts 0.9
    without self compare:
    python functionPoint_Compare.py -ts 0.9
"""
import os
import json
import pandas as pd
import argparse
from collections import defaultdict
from data_utils import build_compare_method, process_excel_data, get_systemFunctionInfo_map, read_work_book, TODAY, logger

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="SequenceMatcher", help="[jieba_set_compare, SequenceMatcher]")
parser.add_argument("--self_compare", action="store_true", default=False, help="whether to self_compare new excel file")
parser.add_argument("-ts", "--threshold", type=float, default=0.5)
args = parser.parse_args()


def self_compare(path, compare_method, threshold=0.5):
    duplicate = []
    _dict = defaultdict(list)
    workbook = read_work_book(path)
    all_system, all_system_info = process_excel_data(workbook)
    
    # build {system:[system_info]} dict
    for system, system_info in zip(all_system, all_system_info):
        _dict[system].append(system_info)
    # compare
    logger.info("new excel file self comparing ....")
    for system in _dict.keys():
        system_info = _dict[system]
        for query_idx in range(len(system_info)):
            for candidate_idx in range(1, len(system_info)):
                if system_info[query_idx] == system_info[candidate_idx]:
                    duplicate.append([system, system_info[query_idx], system_info[candidate_idx], 1.0])
                else:
                    score = compare_method(system_info[query_idx], system_info[candidate_idx])
                    if score >= threshold:
                        duplicate.append([system, system_info[query_idx], system_info[candidate_idx], score])

    self_compare_result = pd.DataFrame(duplicate, columns=["system","system_info","candidate","score"])
    self_compare_result.to_csv(f"./result/{TODAY}_self_compare.csv",index=False, sep=' ', encoding="UTF-8")
    logger.info("new excel file self compared successed")
    

def one_query_recall(query:dict, systemInfo_map:dict, compare_method, threshold=0.5):
    """
    Args: 
        query (dict): {system_name : system_info}
        systemInfo_map (dict): _description_
        compare_method (Compare): the method of compare query and candidate 
        threshold (float, optional): _description_. Defaults to 0.5

    Returns:
        recall_candidate: [system, system_info, candidate]
    """
    recall_candidate = []

    (system, system_info), = query.items()
    function_candidateSet = systemInfo_map[system]
 
    for candidate in function_candidateSet:
        if system_info == candidate:
            recall_candidate.append([system, system_info, candidate, 1.0])
        else:
            score = compare_method(system_info, candidate)
            if score >= threshold:
                recall_candidate.append([system, system_info, candidate, score])
            
    return recall_candidate


def batch_query_recall(systemInfo_map:dict, new_excel_path_lst, compare_method, threshold=0.5):
    batch_query_recall_result = []
    batch_query_data = get_systemFunctionInfo_map(data_path_lst=new_excel_path_lst, path="./new_excel_file/")
    for system in batch_query_data.keys():
        if system in systemInfo_map.keys():
            for info in batch_query_data[system]:
                batch_query_recall_result.extend(one_query_recall({system:info}, batch_query_data, compare_method, threshold))
                
    return batch_query_recall_result


def main(args):
    # build_compare_method
    compare_method = build_compare_method(args)
    # get db.json
    with open("./db_data/db.json", 'r') as json_file:
        function_map = json.load(json_file, encoding="UTF-8")
    # new excel file self compare
    if args.self_compare:
        self_compare("./new_excel_file/63【南京银行】2022年框架评估_数据挖掘平台22年四季度付款(天阳）-wuhg-0116(已确认)(1).xlsx", compare_method, args.threshold)
    # batch_query_recall()
    logger.info(f"Convert batch_query_recall_result to {TODAY}.csv file ....")
    batch_query_recall_result = batch_query_recall(systemInfo_map=function_map, 
                                                   new_excel_path_lst=os.listdir("./new_excel_file/"), 
                                                   compare_method=compare_method,
                                                   threshold=args.threshold) 
    result = pd.DataFrame(batch_query_recall_result, columns=["system","system_info","candidate","score"])
    result.to_csv(f"./result/{TODAY}.csv",index=False, sep=' ', encoding="UTF-8")
    logger.info("Success!")


if __name__ == "__main__":
    main(args)
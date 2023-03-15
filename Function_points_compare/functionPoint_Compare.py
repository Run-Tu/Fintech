import time
import json
import jieba
import pandas as pd
import argparse
from collections import defaultdict
from data_process_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--one_query", action="store_true", default=False, help="whether to run one_query")
parser.add_argument("--batch_query", action="store_true", default=False, help="whether to run batch_query")
parser.add_argument("-ts", "--threshold", type=float, default=0.5)
args = parser.parse_args()


def one_query_recall(query:dict, systemInfo_map:dict, sw_set:set, threshold=0.5):
    """
    Args:
        query (dict): {system_name : system_info}
        systemInfo_map (dict): _description_
        sw_set(set): stop words set
        threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        result (dict): 
    """
    result = defaultdict(list)
    recall_candidate = []

    (system, system_info), = query.items()
    function_candidateSet = systemInfo_map[system]

    def _compare(query, candidate):
        """
            TODO:考虑更新比较方式，如加入规则或使用语义理解的方式
        """
        set_query = set(jieba.lcut(query, cut_all=True)).difference(sw_set)
        set_candidate = set(jieba.lcut(candidate, cut_all=True)).difference(sw_set)
        
        return len(set_query&set_candidate) / (len(set_candidate)+1)
    
    for candidate in function_candidateSet:
        if _compare(system_info, candidate) >= threshold:
            recall_candidate.append(candidate)
            
    result[system] = {system_info: recall_candidate}

    return result


def batch_query_recall(systemInfo_map:dict, sw_set:set, threshold=0.5):
    """
        TODO：考虑优化时间复杂度
        1、先过滤掉停用词【只需要过滤一次】
        2、优先判断是否相同，相同不走分词和复杂判断逻辑
    """
    result = []
    batch_query_data = get_systemFunctionInfo_map(os.path.join("./original_data/", "63【南京银行】2022年框架评估_数据挖掘平台22年四季度付款(天阳）-wuhg-0116(已确认)(1).xlsx"))
    for system in batch_query_data.keys():
        if system in systemInfo_map.keys():
            for info in batch_query_data[system]:
                result.append(one_query_recall({system:info}, systemInfo_map, sw_set, threshold))
    
    return result


def restore_to_csv(result:dict, csv_path="./result_data"):

    pass


if __name__ == "__main__":
    function_map = get_systemFunctionInfo_map(data_path)
    sw_set = load_stopWords()

    if args.one_query:
        query = {"数据挖掘平台":"应用实施类[SEP]新报表平台重构[SEP]新报表平台重构[SEP]报表平台新增机构、角色关系管理查询"}
        start = time.time()
        result = one_query_recall(query, function_map, sw_set, threshold=args.threshold)
        end = time.time()
        print(f"one_query_recall cost time: {end-start}")
        print(json.dumps(result, ensure_ascii=False))
    
    if args.batch_query:
        """
            1、写成文件
            2、重复度展示
        """
        start = time.time()
        result = batch_query_recall(function_map, sw_set, threshold=args.threshold)
        end = time.time()
        print(f"batch_query_recall cost time: {end-start}")
        print(json.dumps(result, ensure_ascii=False))
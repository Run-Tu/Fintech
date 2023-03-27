# 功能点比对使用方法
## 1、在根目录下创建original_data，new_excel_file和db_data三个文件夹
- original_data用于存放原始数据，通过original_data中的excel数据来建数据库
- db_data用于存放通过original_data数据创建的json文件
- new_excel_file文件夹用于存放待比较的excel表
## 2、脚本执行命令
- python create_dbjson.py 创建dbjson文件至db_data文件夹中
- python functionPoint_Compare.py --self_compare -ts 0.9 生成比较后的csv结果文件,通过-ts来设置比较两个功能点相似度的阈值

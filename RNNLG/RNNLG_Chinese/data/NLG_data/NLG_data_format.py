"""
Usage:
python3 NLG_data_format.py

Output:
train.json, valid.json, test.json

Total data will be split into the following parts:
60% - train set
20% - validation set
20% - test set
"""
import csv
import ast
import math
import json, codecs

def search_disease_file(file, disease_name, column):
    for row in file:
        if row[0] == disease_name:
            return ",".join(ast.literal_eval(row[column]))
    return ""


def search_doctor(file, disease_name, division_name):
    for row in file:
        if disease_name != "":
            if disease_name in row[1]:
                return ",".join(ast.literal_eval(row[2]))
        elif division_name != "":
            if division_name in row[0]:
                return ",".join(ast.literal_eval(row[2]))
    return ""


def split_data(json_list, train_json, valid_json, test_json):
    for item in json_list[:math.floor(len(json_list) * 0.6)]:
        train_json.append(item)
    json_list = json_list[math.floor(len(json_list) * 0.6):]
    for item in json_list[:math.floor(len(json_list) * 0.5)]:
        valid_json.append(item)
    for item in json_list[math.floor(len(json_list) * 0.5):]:
        test_json.append(item)


def create_json_list(json_list, select_list, sen1, sen2, sen3):
    item = []
    for list_item in select_list:
        item.append(sen1 + list_item + "')")
        item.append(sen2 + list_item)
        item.append(sen3 + list_item)
        json_list.append(item)
        item = []


def write_to_json(file_name, json_list):
    with open(file_name, "w", encoding='utf-8') as data_write:
        json.dump(json_list, data_write, ensure_ascii=False)
    data_write.close()

def main():

    with open('division.csv', 'r') as div_rf:
        reader = csv.reader(div_rf)
        division_csv = list(reader)
    with open('disease.csv', 'r') as dis_rf:
        reader = csv.reader(dis_rf)
        disease_csv = list(reader)
    div_rf.close()
    dis_rf.close()
    with open('disease_dict.txt', 'r') as f:
        content = f.readlines()
        disease_list = [x.strip() for x in content]
    f.close()
    with open('division_dict.txt', 'r') as f:
        content = f.readlines()
        division_list = [x.strip() for x in content]
    f.close()
    with open('doctor_dict.txt', 'r') as f:
        content = f.readlines()
        doctor_list = [x.strip() for x in content]
    f.close()

    train_json = []
    valid_json = []
    test_json = []

# inform
    json_list = [
        ["inform(slot='intent')", "請問你要做什麼", "請告訴我您需要什麼協助"],
        ["inform(slot='disease')", "請問疾病的名稱", "請告訴我疾病的名稱"],
        ["inform(slot='division')", "請問是哪個科別", "請告訴我科別名稱"],
        ["inform(slot='doctor')", "請問要哪位醫生", "請告訴我醫生名稱"],
        ["inform(slot='time')", "請問要哪天", "請您告訴我可以的日期"],
        ["inform(slot='intent')", "請問你想要做什麼", "請告訴我您需要什麼樣的協助"],
        ["inform(slot='disease')", "請問疾病的名稱是什麼", "請您告訴我疾病的名稱"],
        ["inform(slot='division')", "請問是哪一個科別", "請告訴我科別的名稱"],
        ["inform(slot='doctor')", "請問是哪位醫生", "請告訴我醫生的名稱"],
        ["inform(slot='time')", "請問您要哪一天", "請告訴我時間"],
        ["inform(slot='intent')", "請問您要做什麼", "請問您要什麼協助"],
        ["inform(slot='disease')", "疾病的名稱是什麼", "請問是什麼疾病"],
        ["inform(slot='division')", "請問科別是哪個", "請告訴我是哪個科別"],
        ["inform(slot='doctor')", "請問您要哪一位醫生", "請您告訴我一位醫生的名稱"],
        ["inform(slot='time')", "請問什麼日期可以", "請告訴我方便的日期"]
    ]
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# select intent
    select_intent = ['查詢症狀,查詢科別', '查詢症狀,查詢醫生', '查詢症狀,查詢時刻表', '查詢症狀,掛號',
                     '查詢科別,查詢醫生', '查詢科別,查詢時刻表', '查詢科別,掛號',
                     '查詢醫生,查詢時刻表', '查詢時刻表,掛號', '查詢醫生掛號']
    create_json_list(json_list, select_intent, "select(intent='",
                     "請選擇服務項目,", "請問您想要哪個服務,")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []

    create_json_list(json_list, select_intent, "select(intent='",
                     "請選擇您想要的服務項目,", "請問您想要選擇哪一個服務,")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []

    create_json_list(json_list, select_intent, "select(intent='",
                     "請選擇一個您想要的服務項目,", "您要選擇哪一個服務項目,")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# select disease
    select_disease = []
    for index, item in enumerate(disease_list):
        if 2 < index < len(disease_list)-2:
            select_disease.append(item + "," + disease_list[index+1])

    create_json_list(json_list, select_disease, "select(disease='",
                     "請選擇疾病名稱,", "請問是哪一個疾病,")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# select division
    select_division = []
    for index, item in enumerate(division_list):
        if 2 < index < len(division_list)-2:
            select_division.append(item + "," + division_list[index+1])

    create_json_list(json_list, select_disease, "select(division='",
                     "請選擇科別,", "請問您要選擇哪個科別,")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# select doctor
    select_doctor = []
    for index, item in enumerate(doctor_list):
        if 2 < index < len(doctor_list)-2:
            select_doctor.append(item + "," + doctor_list[index+1])

    create_json_list(json_list, select_doctor, "select(division='",
                     "請選擇醫生,", "請問您要選擇哪位醫生,")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# select time
    select_time = ['106.5.5,106.5.6','106.5.4,106.1.6','106.5.25,106.5.16','106.2.5,106.5.4',
                   '107.5.5,106.5.6','106.9.5,106.5.6','106.5.20,106.7.6','106.5.4,106.3.6']

    create_json_list(json_list, select_time, "select(time='",
                     "請選擇日期,", "請問您要哪天,")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# confirm intent
    intent_list = ['查詢症狀', '查詢科別', '查詢醫生', '查詢時刻表', '掛號']
    create_json_list(json_list, intent_list, "confirm(intent='",
                     "請問您是否是要", "您是不是要")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# confirm disease
    create_json_list(json_list, disease_list, "confirm(disease='",
                     "請問您是不是說", "請問是不是這個疾病：")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# confirm division
    create_json_list(json_list, division_list, "confirm(division='",
                     "請問您是否說", "請問是不是這個科別：")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# confirm doctor
    create_json_list(json_list, doctor_list, "confirm(doctor='",
                     "請問您是否指的是", "請問是不是這位醫生：")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# confirm time
    select_time = ['106.5.5','106.5.4','106.1.6','106.5.25','106.5.16','106.2.5','106.5.4',
                   '107.5.5','106.5.6','106.9.5','106.5.6','106.5.20','106.7.6','106.5.4','106.3.6']
    create_json_list(json_list, select_time, "confirm(time='",
                     "請問是否想要這天：", "請問是不是這個日期：")
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []


# End intent=1
    item = []
    for dis in disease_list:
        result = search_disease_file(disease_csv, dis, 4)
        if result != "":
            item.append("end(intent='1';disease='" + dis + "';results='" + result + "')")
            item.append("幫您查詢症狀,以下是" + dis + "的症狀," + result)
            item.append(dis + "的症狀有" + result)
            json_list.append(item)
            item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# End intent=2
    item = []
    for dis in disease_list:
        result = search_disease_file(disease_csv, dis, 2)
        if result != "":
            item.append("end(intent='2';disease='" + dis + "';results='" + result + "')")
            item.append("已查詢到科別," + dis + "的科別是," + result)
            item.append(dis + "的相關科別是" + result)
            json_list.append(item)
            item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# End intent=3
    item = []
    for dis in disease_list:
        result = search_doctor(division_csv, dis, "")
        if result != "":
            item.append("end(intent='3';disease='" + dis + "';results='" + result + "')")
            item.append("已經幫您查詢到" + dis + "的主治醫生有," + result)
            item.append(dis + "的主治醫師有" + result)
            json_list.append(item)
            item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# End intent=3
    item = []
    for dis in disease_list:
        result = search_doctor(division_csv, dis, "")
        if result != "":
            item.append("end(intent='3';division='眼科';results='" + result + "')")
            item.append("已查詢到眼科的醫師有," + result)
            item.append("眼科的醫生有" + result)
            json_list.append(item)
            item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# End intent=4
    item = []
    for dis in disease_list:
        item.append("end(intent='4';disease='" + dis + "';doctor='胡芳蓉';results='106.5.5,106.5.6')")
        item.append("查詢到" + dis + "胡芳蓉醫師的門診時刻有106.5.5,106.5.6,")
        item.append(dis + "胡芳蓉醫師的門診時刻有106.5.5,106.5.6")
        json_list.append(item)
        item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# End intent=4
    item = []
    for dis in division_list:
        item.append("end(intent='4';division='" + dis + "';doctor='胡芳蓉';results='106.5.5,106.5.6')")
        item.append("幫您查詢到" + dis + "胡芳蓉醫師的門診時刻有106.5.5,106.5.6,")
        item.append(dis + "胡芳蓉醫生門診時間表是106.5.5,106.5.6")
        json_list.append(item)
        item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# End intent=5
    item = []
    for dis in disease_list:
        item.append("end(intent='5';disease='" + dis + "';doctor='胡芳蓉';time='106.5.5')")
        item.append("幫您預約掛號" + dis + "胡芳蓉醫師106.5.5的門診")
        item.append(dis + "胡芳蓉醫師106.5.5的門診已預約好了")
        json_list.append(item)
        item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
    item = []
    for dis in disease_list:
        item.append("end(intent='5';disease='" + dis + "';doctor='胡芳蓉';time='106.5.5')")
        item.append("預約了" + dis + "胡芳蓉醫師106.5.5的門診")
        item.append(dis + "胡芳蓉醫師106.5.5的門診已預約完成")
        json_list.append(item)
        item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
# End intent=5
    item = []
    for dis in division_list:
        item.append("end(intent='5';division='" + dis + "';doctor='胡芳蓉';time='106.5.5')")
        item.append("已幫您掛號" + dis + "胡芳蓉醫師106.5.5的門診")
        item.append(dis + "胡芳蓉醫師106.5.5的門診已掛號完成")
        json_list.append(item)
        item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []
    item = []
    for dis in division_list:
        item.append("end(intent='5';division='" + dis + "';doctor='胡芳蓉';time='106.5.5')")
        item.append("掛了" + dis + "胡芳蓉醫師106.5.5的門診")
        item.append(dis + "胡芳蓉醫師106.5.5的門診已掛好了")
        json_list.append(item)
        item = []
    split_data(json_list, train_json, valid_json, test_json)
    json_list = []

    print(len(json_list))
    print("-----------------")
    print(len(train_json))
    print("-----------------")
    print(len(valid_json))
    print("-----------------")
    print(len(test_json))

    write_to_json("train.json", train_json)
    write_to_json("valid.json", valid_json)
    write_to_json("test.json", test_json)

if __name__ == '__main__':
    main()

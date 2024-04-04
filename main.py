from find_all_possible_ports import find_ports
from nearesr_raw import cal_time_oil_cost,find_routes_pass_by
from geopy.distance import geodesic
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, atan2, degrees


# #############################全局设定########################################
def load_history_routes(filepath):
    """加载历史航线数据。"""
    with open(filepath, "rb") as file:
        fileJson = json.load(file)
    return fileJson

def tensor_to_dataframe(tensor, columns):
    """将Tensor转换为DataFrame。"""
    np_array = tensor.numpy().reshape(-1, tensor.shape[-1])
    df = pd.DataFrame(np_array, columns=columns)
    return df

if __name__ == "__main__":
    # 加载航线数据
    filePath = "data/history_routes_for_each_ship.txt"
    routeFile = load_history_routes(filePath)

    # 参数设置
    maxLen = 325
    getLen = 25
    inputLen = 24
    shipNum = sum(len(value) >= maxLen for value in routeFile.values())

    # 初始化预测数据Tensor
    predictShipData = torch.zeros([shipNum, inputLen, 11])  # 注意：增加shipId,起点和终点的经纬

    # 构建预测数据集
    shipId = 0
    for key, value in routeFile.items():  # 使用key获取MMSI
        if len(value) >= maxLen:
            for startIndex in range(0, len(value), (inputLen - 1) * getLen):
                endIndex = min(startIndex + (inputLen - 1) * getLen, len(value))
                if endIndex - startIndex < (inputLen - 1) * getLen:
                    break
                # 获取起点和终点的经纬度
                start_lat = value[startIndex]["LAT"]
                start_lon = value[startIndex]["LON"]
                destination_lat = value[endIndex]["LAT"]
                destination_lon = value[endIndex]["LON"]

                for positionId in range(inputLen):
                    dataPointIndex = startIndex + positionId * getLen
                    eachData = value[dataPointIndex]

                    if shipId < shipNum:
                        predictShipData[shipId][positionId] = torch.tensor([
                            shipId,  # 存储shipId作为航线的唯一标识
                            eachData["MMSI"], eachData["LAT"], eachData["LON"],
                            eachData["SOG"], eachData["COG"], eachData["Heading"],
                            start_lat, start_lon,  # 添加起始经纬度
                            destination_lat, destination_lon
                        ])
                shipId += 1

    # 转换为DataFrame
    columns = ["ShipId", "MMSI", "LAT", "LON", "SOG", "COG", "Heading", "Start_LAT", "Start_LON","Destination_LAT", "Destination_LON"]
    df = tensor_to_dataframe(predictShipData, columns)

    # 设置随机数种子 TODO:替换成真实数据
    np.random.seed(0)
    size = df.shape[0]
    # 生成进油口流量的假数据并添加到df
    df['IFR'] = np.random.uniform(40, 100, size=(size))
    # 生成出油口流量的假数据并添加到df
    df['OFR'] = np.random.uniform(80, 450, size=(size))
    # 生成吃水信息的假数据并添加到df
    df['draft'] = np.random.uniform(10, 100, size=(size))
    # 新增航次表的信息
    df['start'] =  np.random.uniform()

    # 打印更新后的df以确认
    # print(df.head())

    # 起点、当前位置和预期终点
    start_position = (29.80755,-92.06513)
    current_position = (29.84484, -91.90000)
    expected_destination = (29.60488, -90.97576)

    # #######################筛选经过起点和终点的航线们##########################
    # TODO:判断这些航线的方向
    possible_routes = find_routes_pass_by(df,start_position,expected_destination)
    # #######################找寻所有可能的港口###############################
    # TODO：检查找到的港口数量，循环+检查
    possible_ports = find_ports(possible_routes)
    print(possible_ports)

    # ######################为沿路每一个港口计算时间和油量消耗###################
    # 使用iterrows遍历DataFrame的每一行
    cost = []
    # 提取第一行的'Port_LAT'和'Port_LON'，并将它们作为一个元组
    first_port = (possible_ports.loc[0, 'Port_LAT'], possible_ports.loc[0, 'Port_LON'])
    result = cal_time_oil_cost(possible_routes, start_position, first_port, start_position)
    # 检查函数是否返回了None
    if result is None:
        # 如果函数返回None，为这三个变量赋值None
        partial_path, predicted_fuel_consumption_rate, time ,price= None, None, None,None
    else:
        # 如果函数返回了有效值，则解包这些值
        partial_path, predicted_fuel_consumption_rate, time ,price = result
    # 创建一个字典，将DataFrame和time作为键值对
    cost_dict = {
        'predicted_fuel_consumption_rate': predicted_fuel_consumption_rate,
        'time': time,
        'price':price
    }
    cost.append(cost_dict)
    # 检查DataFrame的行数是否大于1
    if len(possible_ports) > 1:
        # 从第二行开始迭代，这样我们可以安全地引用前一行(i-1)
        for i in range(1, len(possible_ports)):
            # 获取当前行（i）和前一行（i-1）的'Port_LAT'和'Port_LON'
            current_lat_lon = (possible_ports.iloc[i]['Port_LAT'], possible_ports.iloc[i]['Port_LON'])
            previous_lat_lon = (possible_ports.iloc[i - 1]['Port_LAT'], possible_ports.iloc[i - 1]['Port_LON'])

            result = cal_time_oil_cost(possible_routes,previous_lat_lon, current_lat_lon,previous_lat_lon)
            # 检查函数是否返回了None
            if result is None:
                # 如果函数返回None，为这三个变量赋值None
                partial_path, predicted_fuel_consumption_rate, time ,price = None, None, None,None
            else:
                # 如果函数返回了有效值，则解包这些值
                partial_path, predicted_fuel_consumption_rate, time, price = result
            # 创建一个字典，将DataFrame和time作为键值对
            cost_dict = {
                'predicted_fuel_consumption_rate': predicted_fuel_consumption_rate,
                'time': time,
                'price' : price
            }
            cost.append(cost_dict)
    # 还有到终点的
    # 提取最后一行的'Port_LAT'和'Port_LON'，并将它们作为一个元组
    last_port = (possible_ports.iloc[-1]['Port_LAT'], possible_ports.iloc[-1]['Port_LON'])
    result = cal_time_oil_cost(possible_routes, last_port, expected_destination,last_port)
    # 检查函数是否返回了None
    if result is None:
        # 如果函数返回None，为这三个变量赋值None
        partial_path, predicted_fuel_consumption_rate, time ,price= None, None, None,None
    else:
        # 如果函数返回了有效值，则解包这些值
        partial_path, predicted_fuel_consumption_rate, time ,price= result
    # 创建一个字典，将DataFrame和time作为键值对
    cost_dict = {
        'predicted_fuel_consumption_rate': predicted_fuel_consumption_rate,
        'time': time,
        'price' : price
    }
    cost.append(cost_dict)

    # cost包含了每个阶段的时间和能耗
    print(cost)
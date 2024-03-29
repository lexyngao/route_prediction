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


def is_nearby(point_a, point_b, threshold=15):
    """判断两个点是否接近。"""
    distance = geodesic(point_a, point_b).kilometers
    return distance <= threshold

def calculate_bearing(pointA, pointB):
    """
    Calculate the bearing between two points.
    """
    lat1, lon1 = radians(pointA[0]), radians(pointA[1])
    lat2, lon2 = radians(pointB[0]), radians(pointB[1])
    dLon = lon2 - lon1
    x = cos(lat2) * sin(dLon)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    bearing = atan2(x, y)
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def is_towards_destination_with_curve(df, current_index, closest_to_current_point, threshold=10):
    """考虑曲线轨迹，判断点是否朝向预期终点。"""
    current_position = (df.loc[current_index, "LAT"], df.loc[current_index, "LON"])
    # 前一个点和后一个点，用于估计切线方向
    if current_index > 0 and current_index < len(df) - 1:
        prev_position = (df.loc[current_index - 1, "LAT"], df.loc[current_index - 1, "LON"])
        next_position = (df.loc[current_index + 1, "LAT"], df.loc[current_index + 1, "LON"])
        tangent_bearing = calculate_bearing(prev_position, next_position)
        destination_bearing = calculate_bearing(current_position, closest_to_current_point)
        # 比较方向
        bearing_diff = abs(tangent_bearing - destination_bearing)
        # 确保在0-180度内比较
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
        return bearing_diff <= threshold
    else:
        # 如果没有足够的点来估计切线方向，简化处理（或返回False，或使用其他逻辑）
        return False


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
    print(df.head())

    # 起点、当前位置和预期终点
    start_position = (29.8350,-91.98000)
    current_position = (29.77000, -91.78000)
    expected_destination = (29.6664, -91.39036)



    # 筛选同时经过起点和预期终点附近的航线
    df['IsNearStart'] = df.apply(lambda row: is_nearby((row["Start_LAT"], row["Start_LON"]), start_position), axis=1)
    df['IsNearDestination'] = df.apply(
        lambda row: is_nearby((row["Destination_LAT"], row["Destination_LON"]), expected_destination), axis=1)

    # 筛选同时满足两个条件的ShipId
    nearby_ship_ids = df[df['IsNearStart'] & df['IsNearDestination']]['ShipId'].unique()

    # 在这些航线中，选择与当前位置最短距离最小的航线
    df['DistanceToCurrent'] = df.apply(lambda row: geodesic(current_position, (row["LAT"], row["LON"])).kilometers,
                                       axis=1)
    # 顺便计算一下到终点的确切距离
    df['DistanceToDestination'] = df.apply(lambda row: geodesic(expected_destination, (row["LAT"], row["LON"])).kilometers,
                                       axis=1)
    min_distance = float('inf')
    selected_ship_id = None
    for ship_id in nearby_ship_ids:
        min_distance_ship = df[df['ShipId'] == ship_id]['DistanceToCurrent'].min()
        if min_distance_ship < min_distance:
            min_distance = min_distance_ship
            selected_ship_id = ship_id

    # 获取选中航线的所有点
    selected_path = df[df['ShipId'] == selected_ship_id] if selected_ship_id is not None else pd.DataFrame()

    # 绘制地图和航线
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_extent([-100, -80, 20, 40])

    # 绘制所有历史航线
    for index, row in df.iterrows():
        plt.plot(row['LON'], row['LAT'], marker='o', color='blue', markersize=1, transform=ccrs.Geodetic())

    if not selected_path.empty:
        # Step 1: 找到离currentPosition最近的点
        closest_to_current_idx = selected_path['DistanceToCurrent'].idxmin()
        # 检查是否满足方向要求 TODO：更改成循环直到找到符合要求的
        # 应用这个逻辑
        # 注意：这里需要适当修改，因为is_towards_destination_with_curve设计为逐行处理，依赖于索引和数据框
        closest_to_current_point = (df.loc[closest_to_current_idx, "LAT"], df.loc[closest_to_current_idx, "LON"])
        df['IsTowards'] = False
        for index, row in selected_path.iterrows(): #TODO:Debug:这个index和row都还是原来df的
            df.loc[index, 'IsTowards'] = is_towards_destination_with_curve(df, index, closest_to_current_point,
                                                                           threshold=90)
        selected_path = selected_path[df['IsTowards'] == True]
        # Step 2：找到离Destination最近的点
        closest_to_dest_idx = selected_path['DistanceToDestination'].idxmin()
        # # Step 2: 预期目的地附近的点已经通过IsNearby筛选过，这里直接使用
        # near_destination_indices = selected_path[selected_path['IsNearDestination']].index.tolist()
        # # 如果有多个符合条件的点，选择第一个作为示例
        # near_destination_idx = near_destination_indices[0] if near_destination_indices else None
        # 确保两个特定点的索引都有效
        if closest_to_dest_idx and closest_to_current_idx:
            # Step 3: 绘制选中的整条航线
            # for index, row in selected_path.iterrows():
            #     plt.plot(row['LON'], row['LAT'], marker='o', color='red', markersize=1, transform=ccrs.Geodetic())

            # 确定两点之间的部分（注意处理索引可能颠倒的情况）
            if closest_to_current_idx > closest_to_dest_idx:#TODO:这里其实说明方向不一直，应该有错误处理
                partial_path = df.loc[closest_to_dest_idx:closest_to_current_idx]
            else:
                partial_path = df.loc[closest_to_current_idx:closest_to_dest_idx]

            # 绘制这一部分
            plt.plot(partial_path['LON'].to_numpy(), partial_path['LAT'].to_numpy(),
                     marker='o', color='red', markersize=1, transform=ccrs.Geodetic(), label='Highlighted Path')

    # 绘制起点同终点
    plt.plot(current_position[1], current_position[0], marker='*', color='yellow', markersize=2,
             label='Current Position', transform=ccrs.Geodetic())
    plt.plot(expected_destination[1], expected_destination[0], marker='D', color='pink', markersize=2,
             label='Expected Destination', transform=ccrs.Geodetic())

    plt.legend(loc='best')
    plt.title('Ship Route Visualization')
    plt.show()

    partial_path.to_csv('data/nearest_prediction.csv', index=False, encoding='utf-8-sig')

    ###############################燃油消耗的估算(简单线性规划)###################################################################

    # 选择特征变量和目标变量
    X = df[['SOG', 'draft']]  # 特征变量: 速度和吃水深度
    y = df['OFR'] - df['IFR']  # 目标变量: 燃油消耗率

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(X_train, y_train)
    # 在测试集上预测燃油消耗率
    y_pred = model.predict(X_test)

    # 计算并打印均方误差
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # 打印模型系数
    print(f'Coefficients: {model.coef_}')
    # 假设有一个新的速度和吃水深度数据

    # 使用模型进行预测
    predicted_fuel_consumption_rate = model.predict(partial_path[['SOG','draft']])
    print(f'Predicted Fuel Consumption Rate: {predicted_fuel_consumption_rate[0]}')

    ###############################最终输出(简单线性规划)###################################################################
    time = partial_path.shape[0] * 1 #TODO:替换成真正的行间距
    print(f'final :Predicted Fuel Consumption Rate: {predicted_fuel_consumption_rate[0]}; time costing:',time)

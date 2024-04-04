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


def is_nearby(point_a, point_b, threshold=1):
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

def is_towards_destination_with(df, current_index, closest_to_current_point, threshold=90):
    """考虑曲线轨迹，判断点是否朝向预期终点。"""
    current_position = (df.loc[current_index, "LAT"], df.loc[current_index, "LON"])
    # closest_to_current_point的后一个点，用于估计切线方向
    if closest_to_current_point > 0 and closest_to_current_point < len(df) - 1:
        # prev_position = (df.loc[closest_to_current_point - 1, "LAT"], df.loc[closest_to_current_point - 1, "LON"])
        next_position = (df.loc[closest_to_current_point + 1, "LAT"], df.loc[closest_to_current_point + 1, "LON"])
        tangent_bearing = calculate_bearing(closest_to_current_point, next_position)
        destination_bearing = calculate_bearing(current_position, closest_to_current_point)
        # 比较方向
        bearing_diff = abs(tangent_bearing - destination_bearing)
        # 确保在0-180度内比较（abs应该不会吧）
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
        return bearing_diff <= threshold
    else:
        # 如果没有足够的点来估计切线方向，简化处理（或返回False，或使用其他逻辑）
        return False

def is_towards_destination_with_curve(df, current_position, closest_to_current_index, threshold=90):
    """考虑曲线轨迹，判断点是否朝向预期终点。"""
    # 在循环中检查直到找到满足条件的点或达到数据集末尾
    while closest_to_current_index < len(df) - 1:
        closest_to_current_position = (df.loc[closest_to_current_index, "LAT"], df.loc[closest_to_current_index, "LON"])
        next_position = (df.loc[closest_to_current_index + 1, "LAT"], df.loc[closest_to_current_index + 1, "LON"])
        tangent_bearing = calculate_bearing(closest_to_current_position, next_position)
        test_bearing = calculate_bearing(current_position, closest_to_current_position)

        bearing_diff = abs(tangent_bearing - test_bearing)
        # 确保在0-180度内比较
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff

        # 如果bearing_diff小于或等于阈值，返回True，否则继续循环
        if bearing_diff <= threshold:
            return True, closest_to_current_index

        # 移动到下一个点
        closest_to_current_index += 1

    # 如果循环结束没有找到满足条件的点
    return False, None


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


import pandas as pd


def find_routes_pass_by(df, start_position, expected_destination):
    # 筛选同时经过起点和预期终点附近的航线
    df['IsNearStart'] = df.apply(lambda row: is_nearby((row["LAT"], row["LON"]), start_position), axis=1)
    df['IsNearDestination'] = df.apply(
        lambda row: is_nearby((row["LAT"], row["LON"]), expected_destination), axis=1)

    # 筛选出经过起点附近的船舶ID
    near_start_ids = df[df['IsNearStart']]['ShipId'].unique()
    # 筛选出经过目的地附近的船舶ID
    near_destination_ids = df[df['IsNearDestination']]['ShipId'].unique()
    # 找出同时满足两个条件的船舶ID
    nearby_ship_ids = set(near_start_ids).intersection(near_destination_ids)

    # 初始化一个空的DataFrame来收集所有选中的航线点
    selected_paths = pd.DataFrame()
    for ship_id in nearby_ship_ids:
        # 获取选中航线的所有点
        selected_path = df[df['ShipId'] == ship_id] if ship_id is not None else pd.DataFrame()
        # 将当前航线的点添加到selected_paths中
        selected_paths = pd.concat([selected_paths, selected_path], ignore_index=True)

    # 返回所有选中的航线点
    return selected_paths


# if __name__ == "__main__":
def cal_time_oil_cost(df,start_position,expected_destination,current_position):
    min_distance = float('inf')
    min_distance_dest = float('inf')
    selected_ship_id = None
    closest_point_idx = None  # 用于记录最近点的索引
    closest_to_dest_idx = None

    for idx, row in df.iterrows():
        # 计算当前行船舶位置到当前位置的距离
        distance_to_current = geodesic(current_position, (row["LAT"], row["LON"])).kilometers
        # 计算到终点的距离
        distance_to_dest = geodesic(expected_destination, (row["LAT"], row["LON"])).kilometers

        # 检查这个距离是否是目前遇到的最值
        if distance_to_current < min_distance:
            min_distance = distance_to_current
            selected_ship_id = row['ShipId']
            closest_point_idx = idx  # 更新最近点的索引
        if distance_to_dest < min_distance_dest:
            min_distance_dest = distance_to_dest
            closest_to_dest_idx = idx  # 更新最近点的索引
    # 获取选中航线的所有点
    selected_path = df[df['ShipId'] == selected_ship_id] if selected_ship_id is not None else pd.DataFrame()
    # TODO:添加价格预测 这里并不是很合适的位置
    prices_array = np.load('data/pred.npy')
    prices_array = prices_array[0, :, :]
    # 重塑数组
    reshaped_array = prices_array.reshape(-1, 8)
    # 创建 DataFrame
    prices_df = pd.DataFrame(reshaped_array, columns=[f"Prices_{i}" for i in range(1, 9)])
    add_prices_df = pd.concat([selected_path.reset_index(drop=True), prices_df.reset_index(drop=True)], axis=1)

    if not selected_path.empty:
        # Step 1: 找到离currentPosition最近的点
        # 检查至currentPosition满足方向要求
        is_toward,closest_to_current_idx = is_towards_destination_with_curve(df,current_position,closest_point_idx,90)
        if not is_toward:# TODO:添加方向不同的错误提示
            return None
        # Step 2：找到离Destination最近的点
        # 确保两个特定点的索引都有效
        if closest_to_dest_idx and closest_to_current_idx:
            # 确定两点之间的部分（注意处理索引可能颠倒的情况）
            if closest_to_current_idx > closest_to_dest_idx:  # TODO:这里其实说明方向不一致，应该有错误处理
                partial_path = df.loc[closest_to_dest_idx:closest_to_current_idx]
            else:
                partial_path = df.loc[closest_to_current_idx:closest_to_dest_idx]

    # Step 3: 绘制地图和航线
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_extent([-100, -80, 20, 40])

    # 绘制所有经过起点和终点的历史航线
    for index, row in df.iterrows():
        plt.plot(row['LON'], row['LAT'], marker='o', color='blue', markersize=1, transform=ccrs.Geodetic())

    # 绘制最后选择的部分
    plt.plot(partial_path['LON'].to_numpy(), partial_path['LAT'].to_numpy(),
             marker='o', color='red', markersize=1, transform=ccrs.Geodetic(), label='Highlighted Path')

    # 绘制起点+终点
    plt.plot(current_position[1], current_position[0], marker='*', color='yellow', markersize=2,
             label='Current Position', transform=ccrs.Geodetic())
    plt.plot(expected_destination[1], expected_destination[0], marker='D', color='pink', markersize=2,
             label='Expected Destination', transform=ccrs.Geodetic())

    plt.legend(loc='best')
    plt.title('Ship Route Visualization')
    plt.show()

    # partial_path.to_csv('data/nearest_prediction.csv', index=False, encoding='utf-8-sig')
    if selected_path.empty or partial_path.empty:
        return None




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
    price = add_prices_df.iloc[time,-1]
    print(f'final :Predicted Fuel Consumption Rate: {predicted_fuel_consumption_rate[0]}; time costing:',time,'prices:',price)
    #输出线路和油耗
    return partial_path,predicted_fuel_consumption_rate,time,price
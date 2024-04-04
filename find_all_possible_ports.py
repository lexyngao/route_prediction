import json
import torch
import numpy as np
import pandas as pd


# 确保港口DataFrame至少包含'PORT_NAME', 'LATITUDE', 'LONGITUDE', 'FUEL_OIL', 'DIESEL'列
# 定义一个函数，用于找到一条航线上所有可能途径的加油港口
import geopandas as gpd


def find_fuel_ports_for_routes(routes_df, ports_df, max_distance_km=20):
    """
    使用 geopandas 和空间索引优化的版本，对于给定的航线DataFrame，找到所有可能途径的加油港口。

    :param routes_df: 包含航线数据的DataFrame，有'LAT'和'LON'列。
    :param ports_df: 包含港口数据的DataFrame。
    :param max_distance_km: 考虑港口的最大距离（公里）。
    :return: 途径的加油港口列表的DataFrame。
    """
    # 将 ports_df 转换为 GeoDataFrame，并创建空间索引
    gdf_ports = gpd.GeoDataFrame(ports_df, geometry=gpd.points_from_xy(ports_df['LONGITUDE'], ports_df['LATITUDE']))
    gdf_ports.sindex

    # 将 routes_df 转换为 GeoDataFrame
    gdf_routes = gpd.GeoDataFrame(routes_df, geometry=gpd.points_from_xy(routes_df['LON'], routes_df['LAT']))

    # 定义搜索半径（以度为单位），假设 1 度大约等于 111 公里
    search_radius = max_distance_km / 111

    # 用集合来存储已经处理过的港口名称
    processed_ports = set()

    results = []

    # 对每个航线点进行空间查询
    for idx, row in gdf_routes.iterrows():
        # 生成当前点的缓冲区，用于查找在给定距离内的港口
        buffer = row.geometry.buffer(search_radius)

        # 使用空间索引找到潜在的港口
        possible_matches_index = list(gdf_ports.sindex.intersection(buffer.bounds))
        possible_matches = gdf_ports.iloc[possible_matches_index]

        # 筛选真正在缓冲区内的港口
        precise_matches = possible_matches[possible_matches.intersects(buffer)]

        for _, match in precise_matches.iterrows():
            # 如果港口名称已经处理过，则跳过
            if match['PORT_NAME'] in processed_ports:
                continue

            # 将港口名称添加到已处理集合中
            processed_ports.add(match['PORT_NAME'])

            # 将匹配结果和对应的航线点索引一起保存
            results.append([idx, match['PORT_NAME'], match.geometry.y, match.geometry.x])

    # 将结果转换为 DataFrame
    df_results = pd.DataFrame(results, columns=['Route_Index', 'Port_Name', 'Port_LAT', 'Port_LON'])

    return df_results


def find_ports(df):
    # ############################寻找所有港口########################################
    ports_df = pd.read_csv("data/Global_Ports.csv")
    historical_routes = df[['LAT', 'LON']]
    # 筛选出提供燃油油(FUEL_OIL)或柴油(DIESEL)服务的港口
    fuel_ports_df = ports_df[(ports_df['FUEL_OIL'] == 'Y') | (ports_df['DIESEL'] == 'Y')]

    # 应用函数找到所有可能途径的加油港口
    possible_ports_df = find_fuel_ports_for_routes(historical_routes, fuel_ports_df)

    # 按照历史航线的顺序遍历港口
    sorted_results = []
    for idx, group in possible_ports_df.groupby('Route_Index'):
        sorted_group = group.sort_values(by='Port_Name')
        sorted_results.extend(sorted_group.values.tolist())

    sorted_ports_df = pd.DataFrame(sorted_results, columns=['Route_Index', 'Port_Name', 'Port_LAT', 'Port_LON'])

    return sorted_ports_df

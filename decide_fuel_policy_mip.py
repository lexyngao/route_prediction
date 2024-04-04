from mip import Model, xsum, minimize, INTEGER
import numpy as np

# 假设条件
start_fuel_price = 68.0
max_fuel_capacity = 1000

# 提供的港口数据，现在包含预计的燃油消耗率的array
ports_data = [
    {'predicted_fuel_consumption_rate': np.array([27.16267756, 25.59834234, 26.04369587, 29.89541383,
                                                  25.55244982, 27.88844876, 20.8397665, 24.95547082,
                                                  23.7969226, 25.52025615, 27.24581006, 25.16988261,
                                                  25.23598581, 21.79957794, 22.80985682, 25.78624889,
                                                  23.99849957]), 'time': 17, 'price': 66.92329},
    {'predicted_fuel_consumption_rate': np.array([17.90866949, 26.52973181, 29.42141026, 21.87948148,
                                                  26.16022686]), 'time': 5, 'price': 67.31487}
]

# 创建模型
m = Model()

# 决策变量：出发时的燃油量
initial_fuel = m.add_var(var_type=INTEGER, lb=0, ub=max_fuel_capacity)  # 确保初始燃油量非零

# 决策变量：每个港口的加油量
refuel_amounts = [m.add_var(var_type=INTEGER, lb=0, ub=max_fuel_capacity) for _ in ports_data]

# 目标函数：最小化总成本
total_cost = start_fuel_price * initial_fuel + xsum(
    refuel_amounts[i] * ports_data[i]['price'] for i in range(len(ports_data)))
m.objective = minimize(total_cost)

# 初始燃油量
fuel_before_refueling = initial_fuel

# 为每个港口添加约束条件
for i, data in enumerate(ports_data):
    # 计算到达该港口的燃油消耗
    fuel_consumed = np.sum(data['predicted_fuel_consumption_rate'])
    fuel_before_refueling -= fuel_consumed  # 到达港口前的燃油量
    # 确保到达港口前燃油量不为负
    m += fuel_before_refueling >= 0
    # 更新燃油余量以包含加油
    fuel_before_refueling += refuel_amounts[i]
    # 确保加油后燃油量不超过最大容量
    m += fuel_before_refueling <= max_fuel_capacity

# 求解模型
m.optimize()

# 输出结果
if m.num_solutions:
    print(f"Initial Fuel: {initial_fuel.x} units")
    total_fuel_cost = initial_fuel.x * start_fuel_price  # 初始燃油成本
    fuel_left_before_refueling = initial_fuel.x
    for i, data in enumerate(ports_data):
        fuel_consumed = np.sum(data['predicted_fuel_consumption_rate'])
        fuel_left_before_refueling -= fuel_consumed
        print(f"Port {i + 1}: Fuel before refueling: {fuel_left_before_refueling:.2f} units")
        fuel_left_before_refueling += refuel_amounts[i].x
        refuel_cost = refuel_amounts[i].x * data['price']
        print(f"Port {i + 1}: Refueled {refuel_amounts[i].x} units at cost: {refuel_cost:.2f}")
        total_fuel_cost += refuel_cost

    print(f"Total fuel cost: ${total_fuel_cost:.2f}")
else:
    print("No solution found")
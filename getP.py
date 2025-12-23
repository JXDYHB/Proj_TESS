# getP.py
import numpy as np
from math import gcd
from functools import reduce

def calculate_gcd_of_differences(t0_list):
    """
    计算t0列表中每两个时间点差值的最小公约数，并返回每个差值除以最小公约数后的四舍五入结果的平均值，
    以及每个周期与平均值的差距。

    参数:
    t0_list: 包含transit时间中心t0的列表

    返回:
    average: 所有差值除以最小公约数后的四舍五入结果的平均值
    differences: 每个周期与平均值的差距
    """
    # Step 1: 计算每两个t0之间的差值
    diffs = []
    for i in range(1, len(t0_list)):
        diff = t0_list[i] - t0_list[i-1]
        diffs.append(diff)
    
    if len(diffs) == 0:
        # 只有一个transit，返回0
        return 0, []

    # Step 2: 找出所有差值的最小公约数
    def find_gcd(a, b):
        # 将差值转换为整数再计算最大公约数
        a, b = int(round(a * 1e6)), int(round(b * 1e6))  # 将浮点数放大为整数
        return gcd(a, b)

    # 打印每个差值
    print(f"Calculated differences: {diffs}")
    
    min_gcd = reduce(find_gcd, diffs)
    
    # Step 3: 判断是否存在最小公约数，并根据该公约数调整差值
    normalized_diffs = [round(diff / min_gcd) for diff in diffs]

    # 打印标准化后的差值
    print(f"Normalized differences: {normalized_diffs}")

    # Step 4: 计算四舍五入后的平均值
    average = np.mean(normalized_diffs)

    # Step 5: 计算每个周期与平均值的差距
    differences = [abs(diff - average) for diff in normalized_diffs]

    return average, differences
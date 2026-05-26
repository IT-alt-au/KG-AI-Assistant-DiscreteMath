import pandas as pd
import numpy as np
import math

def min_max(data):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

def z_score(data):
    data = np.array(data)
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=0)
    normalized = (data - mean_val) / std_val
    return normalized

def decimalscaler(data):
    data = np.array(data)
    j = math.ceil(math.log10(np.max(np.abs(data))))
    normalized = data / (10 ** j)
    return normalized, j


df = pd.read_excel(r"/Users/liushili/Downloads/Scores.xlsx")
print(df)
print()
cpp_scores = df['C++成绩'].values
java_scores = df['Java成绩'].values
python_scores = df['Python成绩'].values

cpp_normalized = min_max(cpp_scores)
print(f"原始数据: {cpp_scores}")
print(f"标准化后: {cpp_normalized}")
print(f"范围: [{np.min(cpp_normalized):.4f}, {np.max(cpp_normalized):.4f}]")
print()

java_normalized = z_score(java_scores)
print("=== Java成绩的标准差标准化结果 ===")
print(f"原始数据: {java_scores}")
print(f"标准化后: {java_normalized}")
print(f"均值: {np.mean(java_normalized):.6f}")
print(f"标准差: {np.std(java_normalized, ddof=0):.6f}")
print()

# 3. 对Python成绩进行小数定标标准化
python_normalized, j_value = decimalscaler(python_scores)
print("=== Python成绩的小数定标标准化结果 ===")
print(f"原始数据: {python_scores}")
print(f"j值: {j_value}")
print(f"除数: 10^{j_value} = {10 ** j_value}")
print(f"标准化后: {python_normalized}")
print(f"最大绝对值: {np.max(np.abs(python_normalized)):.6f}")
print()

# 创建处理后的结果DataFrame
result_df = df.copy()
result_df['C++成绩_标准化'] = cpp_normalized
result_df['Java成绩_标准化'] = java_normalized
result_df['Python成绩_标准化'] = python_normalized

print("=== 最终处理结果 ===")
print(result_df)
print()

print("=== 结果验证 ===")
print("1. C++成绩最小-最大标准化验证:")
print(f"   最小值应为0: {np.min(cpp_normalized):.6f}")
print(f"   最大值应为1: {np.max(cpp_normalized):.6f}")

print("2. Java成绩标准差标准化验证:")
print(f"   均值应接近0: {np.mean(java_normalized):.6f}")
print(f"   标准差应接近1: {np.std(java_normalized, ddof=0):.6f}")

print("3. Python成绩小数定标标准化验证:")
print(f"   最大绝对值应小于1: {np.max(np.abs(python_normalized)):.6f}")





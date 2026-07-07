#!/usr/bin/env python3      # 定义头文件
# -*- coding: UTF-8 -*-     # 定义编码格式
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, Normalizer
from sklearn.datasets import load_iris
import pandas as pd
# 加载鸢尾花数据集
iris = load_iris()
# 由于数据众多，只以前两行数据作为测试数据
X = iris.data[:2, :]
# 将列表转换为pandas DataFrame
X = pd.DataFrame(X)
# 设置浮点数显示格式为普通格式（小数点后两位）
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 输出数据
print('*' * 10, '鸢尾花数据集特征缩放处理', '*' * 10)
print("原始数据集：", '\n', X)

# StandardScaler()函数：
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
print("StandardScaler()函数结果：", '\n', X_standard)

# MinMaxScaler()函数：
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)
print("MinMaxScaler()函数结果：", '\n', X_minmax)

# MaxAbsScaler()函数：
scaler = MaxAbsScaler()
X_maxabs = scaler.fit_transform(X)
print("MaxAbsScaler()函数结果：", '\n', X_maxabs)

# RobustScaler()函数：
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
print("RobustScaler()函数结果：", '\n', X_robust)

# Normalizer()函数：
scaler = Normalizer(norm='l1')
X_normalizer = scaler.fit_transform(X)
# 将列表转换为pandas DataFrame
X_normalizer = pd.DataFrame(X_normalizer)
# 设置浮点数显示格式为普通格式（小数点后4位）
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print("Normalizer()函数结果：", '\n', X_normalizer)

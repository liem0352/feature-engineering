#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

# 使用本地生成的示例数据，模拟Statlog (Vehicle)数据集的后三列特征
data = {
    'att4': [95.0, 105.0, 100.0, 102.0, 98.0, 101.0, 97.0, 103.0, 104.0, 99.0],
    'att5': [14.0, 16.0, 15.0, 15.5, 14.5, 15.2, 14.8, 15.8, 16.2, 14.9],
    'att6': [1.0, 2.0, 1.5, 1.8, 1.2, 1.7, 1.4, 1.9, 2.1, 1.3]
}
X = pd.DataFrame(data)

# 初始化KBinsDiscretizer对象
kbd1 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_discretize1 = kbd1.fit_transform(X)

kbd2 = KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile')
X_discretize2 = kbd2.fit_transform(X)

kbd3 = KBinsDiscretizer(n_bins=7, encode='onehot-dense', strategy='kmeans')
X_discretize3 = kbd3.fit_transform(X)

print('*'*10, "数据集离散化处理", '*'*10)
print("原始数据：")
print(X)
print("\n离散化后的数据（3个箱子，特征序数编码，均匀分配）：")
print(X_discretize1)
print("\n离散化后的数据（5个箱子，特征独热编码，分位数分配数据）：")
print(X_discretize2.toarray())  # 转换稀疏矩阵为密集矩阵
print("\n离散化后的数据（7个箱子，特征密集独热编码，K-Means算法分配数据）：")
print(X_discretize3)
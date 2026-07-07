#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import fetch_openml

# 获取数据集
# 要获取ID为18的Statlog (Vehicle)数据集
dataset = fetch_openml(data_id=18, as_frame=True)
# 选择有变化的特征列：att4, att5, att6
X = dataset.data.iloc[:10, [3, 4, 5]]  # 使用后三列特征

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

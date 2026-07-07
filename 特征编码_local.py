#!/usr/bin/env python3      # 定义头文件
# -*- coding: UTF-8 -*-     # 定义编码格式
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# 使用本地生成的示例数据，模拟Statlog (Vehicle)数据集
np.random.seed(42)  # 设置随机种子以确保结果可重复

# 生成分类特征
data = {
    'att1': np.random.choice(['a', 'b', 'c', 'd'], size=10),
    'att2': np.random.choice(['x', 'y', 'z'], size=10),
    'att3': np.random.randint(1, 10, size=10),
    'class': np.random.choice(['car', 'bus', 'truck', 'bike'], size=10)
}

data = pd.DataFrame(data)

print('*'*10, "数据集特征编码", '*'*10)
print("数据集：\n", data)

# 使用LabelEncoder将分类数据转换为数值型数据
label_encoder = LabelEncoder()
data['att1_encoded'] = label_encoder.fit_transform(data['att1'])
data['att2_encoded'] = label_encoder.fit_transform(data['att2'])
print("LabelEncoder 标签编码:")
print(data)

# 使用OrdinalEncoder将分类数据转换为数值型数据，并保留原始顺序关系
ordinal_encoder = OrdinalEncoder()
data_frame2 = data[['att1', 'att2']]
data[['att1_ordinal', 'att2_ordinal']] = ordinal_encoder.fit_transform(data_frame2)
print("OrdinalEncoder序号编码:")
print(data)

# 使用OneHotEncoder将分类数据转换为独热编码形式
onehot_encoder = OneHotEncoder(sparse_output=False)  # 设置sparse_output=False以获取密集矩阵
one_hot_encoded = onehot_encoder.fit_transform(data[['class']])
# 创建独热编码的列名
onehot_cols = [f'class_onehot_{i}' for i in range(one_hot_encoded.shape[1])]
# 将独热编码结果添加到数据框中
for i, col in enumerate(onehot_cols):
    data[col] = one_hot_encoded[:, i]
print("OneHotEncoder独热编码:")
print(data)
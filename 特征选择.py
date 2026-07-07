#!/usr/bin/env python3      # 定义头文件
# -*- coding: UTF-8 -*-     # 定义编码格式
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:5]
y = iris.target[:5]
feature_names = iris.feature_names
print('*'*10, "数据集特征选择", '*'*10)
print("原始数据：")
print(X)
print("原始目标数据：")
print(y)
print("转换前的特征数据：")
print(feature_names)
# 使用VarianceThreshold函数删除低方差特征
threshold = VarianceThreshold(threshold=0.03)
X_high_variance = threshold.fit_transform(X)
print("VarianceThreshold结果（方差阈值过滤）：")
print(X_high_variance)
# X_high_variance包含了删除低方差特征后的数据。
# 查看被保留特征
selected_features = threshold.get_support(indices=True)
print("删除的低方差特征:", selected_features)
print("经过特征选择后的数据:", X_high_variance)
print("特征方差:", threshold.variances_)
print("转换前的特征数量:", threshold.n_features_in_)
print("转换后的特征名称:", threshold.get_feature_names_out())
# 使用SelectKBest类选择最佳特征
# 通过卡方检验（chi2）选择了最佳的2个特征。
selector = SelectKBest(chi2, k=2)
X_best_features = selector.fit_transform(X, y)
print("SelectKBest结果（2个最佳特征）：")
print(X_best_features)
print("原始特征数量：", X.shape[1])
print("筛选后的特征数量：", X_best_features.shape[1])
# 通过卡方检验（chi2）选择了最佳的1个特征。
selector = SelectKBest(chi2, k=1)
X_best_features = selector.fit_transform(X, y)
print("SelectKBest结果（1个最佳特征）：")
print(X_best_features)
print("原始特征数量：", X.shape[1])
print("筛选后的特征数量：", X_best_features.shape[1])

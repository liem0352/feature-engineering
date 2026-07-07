# Python 特征工程实践

> 作者：liem

## 项目简介

本项目是 Python 机器学习中特征工程的系列练习，使用 scikit-learn 和 pandas 对经典数据集（鸢尾花、Statlog Vehicle 等）进行特征编码、特征缩放、特征选择、特征离散化及缺失值处理等操作，涵盖了数据预处理的核心技术。

## 文件说明

| 文件 | 内容 | 使用数据集 |
|------|------|-----------|
| `特征编码.py` | LabelEncoder 标签编码、OrdinalEncoder 序号编码、OneHotEncoder 独热编码 | Statlog (Vehicle) 数据集 (openml id=18) |
| `特征缩放.py` | StandardScaler 标准化、MinMaxScaler 归一化、MaxAbsScaler、RobustScaler、Normalizer | 鸢尾花数据集 |
| `特征选择.py` | VarianceThreshold 方差选择、SelectKBest + 卡方检验选择最佳特征 | 鸢尾花数据集 |
| `特征离散化.py` | KBinsDiscretizer 等宽分箱(uniform)、等频分箱(quantile)、KMeans 聚类分箱 | Statlog (Vehicle) 数据集 |
| `missing_values.py` | 缺失值检测（isnull）、删除缺失行（dropna）、均值填充（fillna） | 鸢尾花数据集 |
| `sklearn_imputation.py` | SimpleImputer 均值/中位数/众数填充、KNNImputer K近邻填充 | 鸢尾花数据集 |
| `exercise.py` | 综合练习：完整的数据预处理流程 | - |

## 技术栈

- Python 3
- pandas
- numpy
- scikit-learn

## 核心知识点

### 特征编码
将分类变量转换为数值型，便于机器学习模型处理：
- **LabelEncoder**：将目标标签编码为 0 到 n-1
- **OrdinalEncoder**：将特征编码为序号，保留顺序关系
- **OneHotEncoder**：将分类特征转为独热编码（二进制向量）

### 特征缩放
消除不同特征的量纲影响，提升模型收敛速度：
- **StandardScaler**：去均值并按方差缩放（Z-score 标准化）
- **MinMaxScaler**：缩放到 [0, 1] 区间
- **MaxAbsScaler**：缩放到 [-1, 1] 区间
- **RobustScaler**：基于中位数和四分位数缩放（抗异常值）
- **Normalizer**：按行归一化

### 特征选择
从原始特征中选择最相关的子集，降低维度：
- **VarianceThreshold**：删除低方差特征
- **SelectKBest + chi2**：通过卡方检验选择最佳特征

### 特征离散化
将连续变量转为离散区间：
- **uniform**：等宽分箱
- **quantile**：等频分箱
- **kmeans**：K-Means 聚类分箱

### 缺失值处理
- **检测**：`isnull()` + `sum()`
- **删除**：`dropna()`
- **填充**：`fillna(均值)`、`SimpleImputer`、`KNNImputer`

## 使用方式

```bash
# 安装依赖
pip install pandas numpy scikit-learn

# 运行各个脚本
python 特征编码.py
python 特征缩放.py
python 特征选择.py
python 特征离散化.py
python missing_values.py
python sklearn_imputation.py
```

## 示例输出

运行 `特征缩放.py` 后可以看到不同缩放方法对原始数据的处理结果：

```
原始数据集：
       0     1     2     3
0  5.10  3.50  1.40  0.20
1  4.90  3.00  1.40  0.20

StandardScaler() 函数结果：
       0     1     2     3
0  0.71  0.71  0.71 -0.71
1 -0.71 -0.71 -0.71  0.71
```

## 学习建议

1. 按顺序学习：编码 → 缩放 → 选择 → 离散化 → 缺失值
2. 修改数据集参数，观察不同处理方法的效果
3. 结合实际项目，选择合适的预处理方法

## 许可证

MIT License

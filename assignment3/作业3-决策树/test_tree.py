import pandas as pd

# 1 读取数据
df = pd.read_csv('D:/car_evaluation.csv')
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

# 2 数据预处理，训练集-测试集划分

# 3 模型训练，选用Gini或者Entropy

# 4 模型评估，是否过拟合，是否需要剪枝

# 5 画出决策树

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1 读取数据
# 使用绝对路径确保文件能被正确找到
file_path = os.path.join(os.path.dirname(__file__), 'car_evaluation.csv')
df = pd.read_csv(file_path)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

# 2 数据预处理，训练集-测试集划分
# 将类别特征转换为数值特征
le = LabelEncoder()
for col in col_names:
    df[col] = le.fit_transform(df[col])

# 分离特征和目标变量
X = df.drop('class', axis=1)
y = df['class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3 模型训练，选用Gini或者Entropy
# 使用Gini指数作为分裂准则
model_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
model_gini.fit(X_train, y_train)

# 使用熵作为分裂准则
model_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
model_entropy.fit(X_train, y_train)

# 4 模型评估，是否过拟合，是否需要剪枝
# 评估Gini模型
y_pred_gini = model_gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, y_pred_gini)
print(f"Gini模型准确率: {accuracy_gini:.4f}")

# 评估Entropy模型
y_pred_entropy = model_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
print(f"Entropy模型准确率: {accuracy_entropy:.4f}")

# 计算训练集准确率，检查是否过拟合
train_accuracy_gini = accuracy_score(y_train, model_gini.predict(X_train))
train_accuracy_entropy = accuracy_score(y_train, model_entropy.predict(X_train))
print(f"Gini模型训练集准确率: {train_accuracy_gini:.4f}")
print(f"Entropy模型训练集准确率: {train_accuracy_entropy:.4f}")

# 5 画出决策树
# 画出Gini决策树
plt.figure(figsize=(20, 10))
plot_tree(model_gini, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.title('Decision Tree with Gini Criterion')
plt.savefig('decision_tree_gini.png')
plt.show()

# 画出Entropy决策树
plt.figure(figsize=(20, 10))
plot_tree(model_entropy, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.title('Decision Tree with Entropy Criterion')
plt.savefig('decision_tree_entropy.png')
plt.show()

# 实现剪枝
# 使用成本复杂度剪枝
path = model_gini.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# 训练不同alpha值的模型
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion='gini', ccp_alpha=ccp_alpha, random_state=42)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# 评估剪枝后的模型
print("\n剪枝后模型评估:")
for i, clf in enumerate(clfs):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Alpha={ccp_alphas[i]:.6f}, 准确率={accuracy:.4f}")


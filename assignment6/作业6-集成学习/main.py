import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# 1 读取数据
file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(file_path)

# 分离特征和目标变量
X = df.drop('price_range', axis=1)
y = df['price_range']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2 使用Adaboost分类器训练模型
# 初始化Adaboost分类器
adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)

# 训练模型
adaboost.fit(X_train, y_train)

# 3 模型在测试集的运行结果评估
# 在测试集上进行预测
y_pred = adaboost.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 生成分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性分析
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': adaboost.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\n特征重要性排序:")
print(feature_importance)


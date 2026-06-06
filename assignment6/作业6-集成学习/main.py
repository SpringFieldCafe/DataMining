from pathlib import Path

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# 所有输入、输出文件都放在当前作业 6 文件夹下，避免从其他目录运行时路径出错。
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
REPORT_PATH = BASE_DIR / "adaboost_evaluation.txt"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"


# 1 读取数据
df = pd.read_csv(DATA_PATH)

# price_range 是分类标签，其余列作为手机属性特征。
X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)


# 2 使用 Adaboost 分类器训练模型
adaboost = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42)
adaboost.fit(X_train, y_train)


# 3 模型在测试集的运行结果评估
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": adaboost.feature_importances_,
    }
).sort_values("Importance", ascending=False)

output_text = (
    f"模型准确率: {accuracy:.4f}\n\n"
    "分类报告:\n"
    f"{report}\n"
    "特征重要性排序:\n"
    f"{feature_importance.to_string(index=False)}\n"
)

print(output_text)

REPORT_PATH.write_text(output_text, encoding="utf-8-sig")
feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

print(f"评估报告已保存到: {REPORT_PATH}")
print(f"特征重要性已保存到: {FEATURE_IMPORTANCE_PATH}")

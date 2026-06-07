from pathlib import Path

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# 所有输入、输出文件都放在当前作业 6 文件夹下，避免从其他目录运行时路径出错
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"

REPORT_PATH = BASE_DIR / "adaboost_evaluation.txt"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.csv"
PREDICT_PROBA_PATH = BASE_DIR / "predict_probability_sample.csv"


# 1. 读取数据
df = pd.read_csv(DATA_PATH)

# price_range 是分类标签，其余列作为手机属性特征
X = df.drop("price_range", axis=1)
y = df["price_range"]


# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# 3. 使用 AdaBoost 分类器训练模型
# sklearn 1.6 以后 algorithm 参数已废弃，因此这里不再写 algorithm="SAMME"
adaboost = AdaBoostClassifier(
    n_estimators=100,
    random_state=42,
)

adaboost.fit(X_train, y_train)


# 4. 模型预测
y_pred = adaboost.predict(X_test)


# 5. 重要评价指标
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=[f"真实类别_{c}" for c in adaboost.classes_],
    columns=[f"预测类别_{c}" for c in adaboost.classes_],
)


# 6. 特征重要性
feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": adaboost.feature_importances_,
    }
).sort_values("Importance", ascending=False)


# 7. 预测概率，也就是模型对每个类别的置信度
proba = adaboost.predict_proba(X_test)

proba_df = pd.DataFrame(
    proba,
    columns=[f"类别_{c}_概率" for c in adaboost.classes_],
)

prediction_sample = X_test.reset_index(drop=True).copy()
prediction_sample["真实类别"] = y_test.reset_index(drop=True)
prediction_sample["预测类别"] = y_pred
prediction_sample["预测是否正确"] = prediction_sample["真实类别"] == prediction_sample["预测类别"]

prediction_sample = pd.concat(
    [
        prediction_sample,
        proba_df,
    ],
    axis=1,
)

prediction_sample_head = prediction_sample.head(10)


# 8. 组织输出文本
output_text = (
    "========== AdaBoost 手机价格等级分类实验结果 ==========\n\n"
    f"数据集路径: {DATA_PATH}\n"
    f"样本总数: {len(df)}\n"
    f"训练集样本数: {len(X_train)}\n"
    f"测试集样本数: {len(X_test)}\n"
    f"特征数量: {X.shape[1]}\n"
    f"类别数量: {len(adaboost.classes_)}\n"
    f"类别标签: {list(adaboost.classes_)}\n\n"
    "========== 1. 整体评价指标 ==========\n\n"
    f"Accuracy 准确率: {accuracy:.4f}\n"
    f"Balanced Accuracy 平衡准确率: {balanced_accuracy:.4f}\n\n"
    "说明:\n"
    "Accuracy 表示整体预测正确的比例。\n"
    "Balanced Accuracy 会分别计算每个类别的召回率后再取平均，适合观察多分类任务中各类别是否均衡。\n\n"
    "========== 2. 分类报告 ==========\n\n"
    f"{report}\n"
    "说明:\n"
    "precision 表示预测为某一类的样本中有多少是真的。\n"
    "recall 表示某一类真实样本中有多少被模型找出来，也叫召回率。\n"
    "f1-score 是 precision 和 recall 的综合指标。\n"
    "support 表示测试集中每个类别的真实样本数量。\n\n"
    "========== 3. 混淆矩阵 ==========\n\n"
    f"{cm_df.to_string()}\n\n"
    "说明:\n"
    "混淆矩阵的行表示真实类别，列表示预测类别。\n"
    "对角线上的数字表示预测正确的数量，非对角线上的数字表示预测错误的数量。\n\n"
    "========== 4. 特征重要性 ==========\n\n"
    f"{feature_importance.to_string(index=False)}\n\n"
    "说明:\n"
    "Importance 越大，说明该特征对 AdaBoost 模型判断手机价格等级的影响越大。\n\n"
    "========== 5. 前 10 个测试样本预测概率 ==========\n\n"
    f"{prediction_sample_head.to_string(index=False)}\n\n"
    "说明:\n"
    "类别_0_概率、类别_1_概率、类别_2_概率、类别_3_概率表示模型认为该样本属于对应价格等级的概率。\n"
)


# 9. 终端输出
print(output_text)


# 10. 保存结果文件
REPORT_PATH.write_text(output_text, encoding="utf-8")
feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
cm_df.to_csv(CONFUSION_MATRIX_PATH, encoding="utf-8-sig")
prediction_sample_head.to_csv(PREDICT_PROBA_PATH, index=False, encoding="utf-8-sig")

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# 所有输入、输出文件都放在当前代码所在文件夹，避免从其他目录运行时路径出错
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"

if not DATA_PATH.exists():
    csv_files = sorted(
        p for p in BASE_DIR.glob("*.csv")
        if p.name not in {
            "classification_report.csv",
            "confusion_matrix.csv",
            "feature_importance.csv",
            "test_predictions.csv",
        }
    )
    if not csv_files:
        raise FileNotFoundError("没有在代码同级目录下找到数据表，请把 data.csv 和 main.py 放在同一文件夹。")
    DATA_PATH = csv_files[0]

REPORT_PATH = BASE_DIR / "classification_report.csv"
CONFUSION_PATH = BASE_DIR / "confusion_matrix.csv"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"
PRED_PATH = BASE_DIR / "test_predictions.csv"
SUMMARY_PATH = BASE_DIR / "summary_metrics.txt"
MODEL_COMPARE_PATH = BASE_DIR / "ensemble_model_comparison.csv"
RF_IMPORTANCE_PATH = BASE_DIR / "random_forest_feature_importance.csv"
ENSEMBLE_SUMMARY_PATH = BASE_DIR / "ensemble_summary.txt"


def out(text: str) -> None:
    """使用 sys 模块输出关键结果。"""
    sys.stdout.write(text + "\n")


# 1 读取数据
df = pd.read_csv(DATA_PATH)
target_col = "price_range" if "price_range" in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# 缺失值做简单填充，保证模型可以正常训练
for col in X.columns:
    if X[col].isna().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0])
X = pd.get_dummies(X, drop_first=False)

stratify_y = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_y,
)

# 2 使用 AdaBoost 分类器训练模型
model = AdaBoostClassifier(
    n_estimators=120,
    learning_rate=0.8,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3 模型在测试集的运行结果评估
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True, zero_division=0)
).T
labels = sorted(y.unique())
confusion_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=labels),
    index=[f"真实_{label}" for label in labels],
    columns=[f"预测_{label}" for label in labels],
)
importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
pred_df = pd.DataFrame({"true_label": y_test, "pred_label": y_pred}, index=y_test.index).sort_index()

report_df.to_csv(REPORT_PATH, encoding="utf-8-sig")
confusion_df.to_csv(CONFUSION_PATH, encoding="utf-8-sig")
importance_df.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
pred_df.to_csv(PRED_PATH, encoding="utf-8-sig")

summary_lines = [
    f"数据文件: {DATA_PATH.name}",
    f"目标变量: {target_col}",
    f"总样本数: {len(df)}",
    f"特征数量: {X.shape[1]}",
    f"训练集样本数: {len(X_train)}",
    f"测试集样本数: {len(X_test)}",
    f"AdaBoost 弱学习器数量 n_estimators: {model.n_estimators}",
    f"AdaBoost 学习率 learning_rate: {model.learning_rate}",
    f"准确率 accuracy: {accuracy:.4f}",
    f"宏平均精确率 macro_precision: {precision_macro:.4f}",
    f"宏平均召回率 macro_recall: {recall_macro:.4f}",
    f"宏平均 F1 macro_f1: {f1_macro:.4f}",
    f"加权精确率 weighted_precision: {precision_weighted:.4f}",
    f"加权召回率 weighted_recall: {recall_weighted:.4f}",
    f"加权 F1 weighted_f1: {f1_weighted:.4f}",
    f"分类报告: {REPORT_PATH.name}",
    f"混淆矩阵: {CONFUSION_PATH.name}",
    f"特征重要度: {IMPORTANCE_PATH.name}",
    f"测试集预测结果: {PRED_PATH.name}",
]
SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

for line in summary_lines:
    out(line)

out("\n各类别 precision / recall / f1-score / support 已保存到 classification_report.csv。")
out("重要特征排名前 10:")
for _, row in importance_df.head(10).iterrows():
    out(f"{row['feature']}: {row['importance']:.4f}")


# 4 集成学习扩展实验：多模型对比
ensemble_models = {
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "AdaBoost": model,
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}

compare_rows = []
for model_name, clf in ensemble_models.items():
    if model_name != "AdaBoost":
        clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    compare_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
    )

compare_df = pd.DataFrame(compare_rows).sort_values(
    ["accuracy", "f1_score"],
    ascending=False,
)
compare_df.to_csv(MODEL_COMPARE_PATH, index=False, encoding="utf-8-sig")

rf_model = ensemble_models["RandomForest"]
rf_importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": rf_model.feature_importances_,
    }
).sort_values("importance", ascending=False)
rf_importance_df.to_csv(RF_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

best = compare_df.iloc[0]
ensemble_summary_lines = [
    "集成学习模型对比结果:",
    compare_df.to_string(index=False),
    "",
    f"最优模型: {best['model']}",
    f"最优模型 accuracy: {best['accuracy']:.4f}",
    f"最优模型 weighted_f1: {best['f1_score']:.4f}",
    "原因分析: 在同一训练集和测试集划分下，最优模型能更好地综合多个弱学习器的判断，因此整体分类误差更低。",
    f"模型对比表: {MODEL_COMPARE_PATH.name}",
    f"随机森林特征重要度: {RF_IMPORTANCE_PATH.name}",
]
ENSEMBLE_SUMMARY_PATH.write_text("\n".join(ensemble_summary_lines), encoding="utf-8")

out("\n集成学习扩展对比:")
out(compare_df.to_string(index=False))
out(f"最优模型: {best['model']}，accuracy={best['accuracy']:.4f}，weighted_f1={best['f1_score']:.4f}")
out(f"模型对比表已保存到: {MODEL_COMPARE_PATH.name}")
out(f"随机森林特征重要度已保存到: {RF_IMPORTANCE_PATH.name}")

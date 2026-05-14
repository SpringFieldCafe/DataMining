import os
import sys

import matplotlib
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1 最小支持度 min_support=0.00030
# 2 最小置信度 min_confidence=0.05
# 3 最小 Lift 值 min_lift=3
# 4 关联规则：限制前项和后项均为单一项的结果
# 5 列出所有规则，以及规则数量
MIN_SUPPORT = 0.00030
MIN_CONFIDENCE = 0.05
MIN_LIFT = 3

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "Groceries_dataset.csv")
RULES_CSV_PATH = os.path.join(BASE_DIR, "association_rules.csv")
LIFT_PLOT_PATH = os.path.join(BASE_DIR, "rules_lift.png")
SCATTER_PLOT_PATH = os.path.join(BASE_DIR, "rules_support_confidence_lift.png")


def write_line(text=""):
    """Use sys.stdout.write to print one line."""
    sys.stdout.write(f"{text}\n")


def itemset_to_text(itemset):
    return ", ".join(sorted(list(itemset)))


def build_transactions(data):
    # 同一会员同一天购买的商品视为一条购物篮记录
    baskets = (
        data.groupby(["Member_number", "Date"])["itemDescription"]
        .apply(list)
        .reset_index(name="items")
    )

    encoder = TransactionEncoder()
    encoded = encoder.fit_transform(baskets["items"])
    return pd.DataFrame(encoded, columns=encoder.columns_)


def mine_rules(transaction_df):
    frequent_itemsets = apriori(
        transaction_df,
        min_support=MIN_SUPPORT,
        use_colnames=True,
    )
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )

    # 只保留“单一前项 -> 单一后项”，并筛选 lift
    rules = rules[
        (rules["antecedents"].apply(len) == 1)
        & (rules["consequents"].apply(len) == 1)
        & (rules["lift"] >= MIN_LIFT)
    ].copy()

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    rules["antecedent"] = rules["antecedents"].apply(itemset_to_text)
    rules["consequent"] = rules["consequents"].apply(itemset_to_text)
    rules["rule"] = rules["antecedent"] + " -> " + rules["consequent"]
    return rules


def save_rules_csv(rules):
    output_columns = [
        "rule",
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
    ]
    rules[output_columns].to_csv(RULES_CSV_PATH, index=False, encoding="utf-8-sig")


def save_plots(rules):
    if rules.empty:
        return

    top_rules = rules.head(10).sort_values("lift")

    plt.figure(figsize=(10, 6))
    plt.barh(top_rules["rule"], top_rules["lift"], color="#4C78A8")
    plt.xlabel("Lift")
    plt.ylabel("Association rule")
    plt.title("Top Association Rules by Lift")
    plt.tight_layout()
    plt.savefig(LIFT_PLOT_PATH, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        s=80,
        alpha=0.85,
    )
    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Support, Confidence and Lift of Rules")
    plt.tight_layout()
    plt.savefig(SCATTER_PLOT_PATH, dpi=200)
    plt.close()


def print_rules(rules):
    write_line("关联规则挖掘结果")
    write_line(f"最小支持度: {MIN_SUPPORT}")
    write_line(f"最小置信度: {MIN_CONFIDENCE}")
    write_line(f"最小 Lift 值: {MIN_LIFT}")
    write_line(f"规则数量: {len(rules)}")
    write_line()

    if rules.empty:
        write_line("未找到满足条件的规则。")
        return

    write_line("所有规则如下：")
    for index, row in rules.iterrows():
        write_line(f"规则 {index + 1}: {row['rule']}")
        write_line(f"  support: {row['support']:.6f}")
        write_line(f"  confidence: {row['confidence']:.4f}")
        write_line(f"  lift: {row['lift']:.4f}")
        write_line()


def main():
    data = pd.read_csv(DATA_PATH)
    data["itemDescription"] = data["itemDescription"].str.strip()
    transaction_df = build_transactions(data)
    rules = mine_rules(transaction_df)

    save_rules_csv(rules)
    save_plots(rules)
    print_rules(rules)

    write_line(f"规则 CSV 已保存到: {RULES_CSV_PATH}")
    if rules.empty:
        write_line("没有生成图片，因为没有满足条件的规则。")
    else:
        write_line(f"Lift 柱状图已保存到: {LIFT_PLOT_PATH}")
        write_line(f"支持度-置信度-Lift 散点图已保存到: {SCATTER_PLOT_PATH}")


if __name__ == "__main__":
    main()

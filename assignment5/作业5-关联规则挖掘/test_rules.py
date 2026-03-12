import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

# 1 读取数据
file_path = os.path.join(os.path.dirname(__file__), 'Groceries_dataset.csv')
df = pd.read_csv(file_path)

# 数据预处理：将数据转换为适合关联规则挖掘的格式
# 按Member_number和Date分组，将每个购物篮中的商品合并为一个列表
basket = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index(name='items')

# 将购物篮数据转换为one-hot编码
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()
transaction_encoded = transaction_encoder.fit_transform(basket['items'])
transaction_df = pd.DataFrame(transaction_encoded, columns=transaction_encoder.columns_)

# 2 使用Apriori算法挖掘频繁项集
min_support = 0.00030
frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)

# 3 生成关联规则
min_confidence = 0.05
min_lift = 3
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# 4 过滤规则：限制前项和后项均为单一一项的结果
rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]

# 5 过滤Lift值
rules = rules[rules['lift'] >= min_lift]

# 6 按lift值排序
rules = rules.sort_values('lift', ascending=False)

# 7 列出所有规则
print("关联规则挖掘结果：")
print(f"总规则数量：{len(rules)}")
print("\n前10条规则（按Lift值排序）：")
for i, row in rules.head(10).iterrows():
    antecedent = list(row['antecedents'])[0]
    consequent = list(row['consequents'])[0]
    support = row['support']
    confidence = row['confidence']
    lift = row['lift']
    print(f"规则 {i+1}: {antecedent} → {consequent}")
    print(f"  支持度: {support:.6f}, 置信度: {confidence:.4f}, Lift: {lift:.4f}")
    print()

# 保存所有规则到文件
rules.to_csv('association_rules.csv', index=False)
print("\n所有规则已保存到 association_rules.csv 文件")


import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 读取数据：csv 文件和 cluster.py 放在同一文件夹下
BASE_DIR = Path(__file__).resolve().parent

# 兼容常见命名：mall_customer.csv / Mall_Customers.csv / 用户上传文件名等
candidate_files = [
    BASE_DIR / 'mall_customer.csv',
    BASE_DIR / 'Mall_Customers.csv',
    BASE_DIR / 'Mall_Customers(2).csv',
]

file_path = None
for p in candidate_files:
    if p.exists():
        file_path = p
        break

# 如果上面的固定文件名都没有找到，就自动寻找当前文件夹下包含 mall 和 customer 的 csv
if file_path is None:
    csv_files = list(BASE_DIR.glob('*.csv'))
    for p in csv_files:
        name = p.name.lower()
        if 'mall' in name and 'customer' in name:
            file_path = p
            break

if file_path is None:
    raise FileNotFoundError('没有在当前文件夹下找到 mall_customer.csv / Mall_Customers.csv，请确认 csv 和 cluster.py 在同一文件夹。')

df = pd.read_csv(file_path)
print(f'读取数据文件：{file_path.name}')
print(df.head())


# 通用函数：使用肘部法则观察 K 的取值
def plot_elbow(X, title, save_name):
    inertia = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',     # 2、K-Means++ 优化起始中心点
            max_iter=300,         # 3、终止条件之一：最大迭代次数
            tol=1e-4,             # 3、终止条件之一：中心点变化小于该阈值则停止
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title(title)
    plt.xlabel('Number of clusters K')
    plt.ylabel('Inertia')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE_DIR / save_name, dpi=300)


# 通用函数：二维聚类图
def plot_2d_cluster(X, k, x_label, y_label, title, save_name):
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',     # 2、K-Means++ 优化起始中心点
        max_iter=300,         # 3、终止条件之一：最大迭代次数
        tol=1e-4,             # 3、终止条件之一：中心点变化小于该阈值则停止
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=250, label='Centroids')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(BASE_DIR / save_name, dpi=300)


# 通用函数：三维聚类图
def plot_3d_cluster(X, k, x_label, y_label, z_label, title, save_name):
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',     # 2、K-Means++ 优化起始中心点
        max_iter=300,         # 3、终止条件之一：最大迭代次数
        tol=1e-4,             # 3、终止条件之一：中心点变化小于该阈值则停止
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='rainbow', s=50)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', marker='*', s=250, label='Centroids')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / save_name, dpi=300)


# Segmentation using Age and spending Score
# 1、K-Means中，首先确定K的取值
# 使用肘部法则观察，Age + Spending Score 这里取 K = 4
X1 = df[['Age', 'Spending Score (1-100)']].values
plot_elbow(X1, 'Elbow Method: Age and Spending Score', 'elbow_age_spending.png')

# 2、K-Means中，确定起始中心点（提示：K-Means++优化）
# 3、K-Means中，确定算法的终止条件
# 4、画出聚类图，同一类的点颜色相同，不同类的点颜色不同
plot_2d_cluster(
    X1,
    k=4,
    x_label='Age',
    y_label='Spending Score (1-100)',
    title='Customer Segmentation: Age and Spending Score',
    save_name='cluster_age_spending.png'
)


# Segmentation using Annual Income and Spending Score
# 使用肘部法则观察，Annual Income + Spending Score 这里取 K = 5
X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
plot_elbow(X2, 'Elbow Method: Annual Income and Spending Score', 'elbow_income_spending.png')
plot_2d_cluster(
    X2,
    k=5,
    x_label='Annual Income (k$)',
    y_label='Spending Score (1-100)',
    title='Customer Segmentation: Annual Income and Spending Score',
    save_name='cluster_income_spending.png'
)


# Segmentation using Age , Annual Income and Spending Score
# 使用肘部法则观察，Age + Annual Income + Spending Score 这里取 K = 6
X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
plot_elbow(X3, 'Elbow Method: Age, Annual Income and Spending Score', 'elbow_age_income_spending.png')
plot_3d_cluster(
    X3,
    k=6,
    x_label='Age',
    y_label='Annual Income (k$)',
    z_label='Spending Score (1-100)',
    title='Customer Segmentation: Age, Annual Income and Spending Score',
    save_name='cluster_age_income_spending.png'
)

print('已生成三个聚类结果图：')
print('1. cluster_age_spending.png')
print('2. cluster_income_spending.png')
print('3. cluster_age_income_spending.png')
print('另外生成了三个肘部法则图，用于说明 K 的取值依据。')

plt.show()

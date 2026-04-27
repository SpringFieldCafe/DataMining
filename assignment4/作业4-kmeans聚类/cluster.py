import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# 读取数据
file_path = os.path.join(os.path.dirname(__file__), 'Mall_Customers.csv')
df = pd.read_csv(file_path)


def elbow_method(X, title, save_name):
    """
    1、确定 K 的取值：使用肘部法则
    """
    inertia = []

    # 这里从 1 到 10 尝试不同的 K
    for k in range(1, 11):
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',   # 2、K-Means++ 初始化中心点
            max_iter=300,       # 3、终止条件之一：最大迭代次数
            tol=1e-4,           # 3、终止条件之二：中心点变化小于 tol
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # 画肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title(title)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(save_name)
    plt.show(block=False)

    return inertia


def plot_2d_clusters(X, y_kmeans, kmeans, optimal_k, colors, title, xlabel, ylabel, save_name):
    """
    4、画二维聚类图
    同一类颜色相同，不同类颜色不同
    """
    plt.figure(figsize=(10, 6))

    for i in range(optimal_k):
        plt.scatter(
            X[y_kmeans == i, 0],
            X[y_kmeans == i, 1],
            s=80,
            c=colors[i],
            label=f'Cluster {i + 1}'
        )

    # 聚类中心
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=250,
        c='yellow',
        marker='X',
        edgecolors='black',
        label='Centroids'
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)
    plt.show(block=False)


# =========================================================
# Segmentation using Age and Spending Score
# =========================================================
print("Segmentation using Age and Spending Score:")
X1 = df[['Age', 'Spending Score (1-100)']].values

# 1、确定 K 的取值
inertia1 = elbow_method(
    X1,
    'Elbow Method for Optimal K (Age vs Spending Score)',
    'elbow_age_spending.png'
)


optimal_k1 = 4

# 2、K-Means++ 优化起始中心点
# 3、终止条件：达到 max_iter=300 或中心点变化小于 tol=1e-4
kmeans1 = KMeans(
    n_clusters=optimal_k1,
    init='k-means++',
    max_iter=300,
    tol=1e-4,
    random_state=42,
    n_init=10
)
y_kmeans1 = kmeans1.fit_predict(X1)

# 4、画出聚类图
colors1 = ['red', 'blue', 'green', 'purple']
plot_2d_clusters(
    X1,
    y_kmeans1,
    kmeans1,
    optimal_k1,
    colors1,
    'Segmentation using Age and Spending Score',
    'Age',
    'Spending Score (1-100)',
    'cluster_age_spending.png'
)


# =========================================================
# Segmentation using Annual Income and Spending Score
# =========================================================
print("\nSegmentation using Annual Income and Spending Score:")
X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 1、确定 K 的取值
inertia2 = elbow_method(
    X2,
    'Elbow Method for Optimal K (Income vs Spending Score)',
    'elbow_income_spending.png'
)

# 根据肘部图选择 K
optimal_k2 = 5

# 2、K-Means++ 优化起始中心点
# 3、终止条件：达到 max_iter=300 或中心点变化小于 tol=1e-4
kmeans2 = KMeans(
    n_clusters=optimal_k2,
    init='k-means++',
    max_iter=300,
    tol=1e-4,
    random_state=42,
    n_init=10
)
y_kmeans2 = kmeans2.fit_predict(X2)

# 4、画出聚类图
colors2 = ['red', 'blue', 'green', 'purple', 'orange']
plot_2d_clusters(
    X2,
    y_kmeans2,
    kmeans2,
    optimal_k2,
    colors2,
    'Segmentation using Annual Income and Spending Score',
    'Annual Income (k$)',
    'Spending Score (1-100)',
    'cluster_income_spending.png'
)


# =========================================================
# Segmentation using Age, Annual Income and Spending Score
# =========================================================
print("\nSegmentation using Age, Annual Income and Spending Score:")
X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# 1、确定 K 的取值
inertia3 = elbow_method(
    X3,
    'Elbow Method for Optimal K (Age, Income, Spending Score)',
    'elbow_age_income_spending.png'
)

# 根据肘部图选择 K
optimal_k3 = 5

# 2、K-Means++ 优化起始中心点
# 3、终止条件：达到 max_iter=300 或中心点变化小于 tol=1e-4
kmeans3 = KMeans(
    n_clusters=optimal_k3,
    init='k-means++',
    max_iter=300,
    tol=1e-4,
    random_state=42,
    n_init=10
)
y_kmeans3 = kmeans3.fit_predict(X3)

# 4、画出 3D 聚类图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors3 = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(optimal_k3):
    ax.scatter(
        X3[y_kmeans3 == i, 0],
        X3[y_kmeans3 == i, 1],
        X3[y_kmeans3 == i, 2],
        s=80,
        c=colors3[i],
        label=f'Cluster {i + 1}'
    )

# 聚类中心
ax.scatter(
    kmeans3.cluster_centers_[:, 0],
    kmeans3.cluster_centers_[:, 1],
    kmeans3.cluster_centers_[:, 2],
    s=250,
    c='yellow',
    marker='X',
    edgecolors='black',
    label='Centroids'
)

ax.set_title('Segmentation using Age, Annual Income and Spending Score')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()
plt.savefig('cluster_age_income_spending.png')
plt.show(block=True)
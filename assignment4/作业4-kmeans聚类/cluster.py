import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# 读取数据
file_path = os.path.join(os.path.dirname(__file__), 'Mall_Customers.csv')
df = pd.read_csv(file_path)

# Segmentation using Age and spending Score
print("Segmentation using Age and Spending Score:")
X1 = df[['Age', 'Spending Score (1-100)']].values

# 1、确定K的取值（肘部法则）
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

# 画出肘部图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K (Age vs Spending Score)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_age_spending.png')
plt.show()

# 2、使用K-Means++优化起始中心点，3、确定算法的终止条件
optimal_k = 4  # 根据肘部图选择
kmeans1 = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans1 = kmeans1.fit_predict(X1)

# 4、画出聚类图
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple']
for i in range(optimal_k):
    plt.scatter(X1[y_kmeans1 == i, 0], X1[y_kmeans1 == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers (Age vs Spending Score)')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('cluster_age_spending.png')
plt.show()

# Segmentation using Annual Income and Spending Score
print("\nSegmentation using Annual Income and Spending Score:")
X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 确定K的取值（肘部法则）
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

# 画出肘部图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K (Income vs Spending Score)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_income_spending.png')
plt.show()

# 使用K-Means++优化起始中心点
optimal_k = 5  # 根据肘部图选择
kmeans2 = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans2 = kmeans2.fit_predict(X2)

# 画出聚类图
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(optimal_k):
    plt.scatter(X2[y_kmeans2 == i, 0], X2[y_kmeans2 == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers (Income vs Spending Score)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('cluster_income_spending.png')
plt.show()

# Segmentation using Age, Annual Income and Spending Score
print("\nSegmentation using Age, Annual Income and Spending Score:")
X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# 确定K的取值（肘部法则）
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)

# 画出肘部图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K (Age, Income, Spending Score)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_age_income_spending.png')
plt.show()

# 使用K-Means++优化起始中心点
optimal_k = 5  # 根据肘部图选择
kmeans3 = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans3 = kmeans3.fit_predict(X3)

# 画出3D聚类图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(optimal_k):
    ax.scatter(X3[y_kmeans3 == i, 0], X3[y_kmeans3 == i, 1], X3[y_kmeans3 == i, 2], s=100, c=colors[i], label=f'Cluster {i+1}')
ax.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], kmeans3.cluster_centers_[:, 2], s=300, c='yellow', label='Centroids')
ax.set_title('Clusters of customers (Age, Income, Spending Score)')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()
plt.savefig('cluster_age_income_spending.png')
plt.show()


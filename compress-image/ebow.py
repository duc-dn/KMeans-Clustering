import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import math
from scipy.spatial.distance import cdist

K = 10


img = plt.imread("./a.jpg")

width = img.shape[1]
height = img.shape[0]

img = img.reshape(width * height, 3)


def visualization_elbow(dataset):
    losses = []
    for i in range(1, K):
        # 1.  Huấn luyện với số cụm = i
        kmeans_i = KMeans(n_clusters=i, random_state=0)
        kmeans_i.fit(dataset)
        # 2. Tính _hàm biến dạng_
        # 2.1. Khoảng cách tới toàn bộ centroids
        d2centroids = cdist(dataset, kmeans_i.cluster_centers_, 'euclidean') # shape (n, k)
        # 2.2. Khoảng cách tới centroid gần nhất
        min_distance = np.min(d2centroids, axis=1) # shape (n)
        loss = np.sum(min_distance)
        losses.append(loss)
    return losses

losses = visualization_elbow(img)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, K), losses, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import math
from scipy.spatial.distance import cdist


img = plt.imread("./a.jpg")

width = img.shape[1]
height = img.shape[0]
print(f"width, height of image: {width} {height}")


img = img.reshape(width * height, 3)
kmeans = KMeans(n_clusters=3).fit(img)
clusters = kmeans.cluster_centers_

print(f"cluster: {clusters}")

labels = kmeans.predict(img)

new_img = np.zeros_like(img)
for pixel in range(0, len(new_img)):
    new_img[pixel] = clusters[labels[pixel]]

new_img = new_img.reshape(height, width, 3)

plt.imshow(new_img)
plt.show()
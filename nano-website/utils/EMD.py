import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


# ===============================================
# Load feature vectors
# ===============================================

df = pd.read_csv(r"D:\Programs\VisualStudio\Repos\nano-website\results-1-5.csv", sep=";")

image_names = df.iloc[:, 0].values
features = df.iloc[:, 1:-1].to_numpy(dtype=float)

# ===============================================
# Normalize each histogram separately
# ===============================================

# # Histogram sizes
# sizes = [19, 25, 95, 95, 95]

# start = 0
# features_norm = np.zeros_like(features)

# for size in sizes:

#     end = start + size

#     hist = features[:, start:end]

#     sums = hist.sum(axis=1, keepdims=True)
#     sums[sums == 0] = 1

#     features_norm[:, start:end] = hist / sums

#     start = end

# ===============================================
# Common X-axis for the "large histogram"
# ===============================================

positions = np.concatenate([
    np.arange(19),
    1000 + np.arange(25),
    3000 + np.arange(95),
    6000 + np.arange(95),
    9000 + np.arange(94)
])

# ===============================================
# Wasserstein distance matrix
# ===============================================

n = len(image_names)

distance_matrix = np.zeros((n, n))

for i in range(n):

    for j in range(i, n):

        d = wasserstein_distance(
            positions,
            positions,
            features[i],
            features[j]
        )

        distance_matrix[i, j] = d
        distance_matrix[j, i] = d

# ===============================================
# Save matrix
# ===============================================

distance_df = pd.DataFrame(
    distance_matrix,
    index=image_names,
    columns=image_names
)

# ===============================================
# Plot
# ===============================================

plt.figure(figsize=(12, 10))

plt.imshow(distance_matrix, cmap="viridis")

plt.colorbar(label="Wasserstein distance")

plt.xticks(
    np.arange(n),
    image_names,
    rotation=90,
    fontsize=7
)

plt.yticks(
    np.arange(n),
    image_names,
    fontsize=7
)

plt.tight_layout()

plt.show()
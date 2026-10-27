# prep_cloud_assets.py
import numpy as np

# Load full test arrays
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Keep a balanced slice of 500 beats across all 5 classes for the cloud demo
cloud_indices = []
for c in range(5):
    idxs = np.where(y_test == c)[0]
    cloud_indices.extend(idxs[:100]) # 100 of each class

cloud_X = X_test[cloud_indices]
cloud_y = y_test[cloud_indices]

np.save('sample_X_test.npy', cloud_X)
np.save('sample_y_test.npy', cloud_y)
print(f"Saved lightweight cloud test set: {cloud_X.shape}")
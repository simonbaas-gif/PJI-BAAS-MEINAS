import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from random import randint
# from sklearn.model_selection import train_test_split


targets = []
features = []

files = glob.glob('train/*.jpg')

for file in files:
    features.append(np.array(Image.open(file).resize((75, 75))))
    target = [1, 0] if "cat" in file else [0, 1]
    targets.append(target)



features = np.array(features)
targets = np.array(targets)

print("features shape", features.shape)
print("Targets shape", targets.shape)

for a in [randint(0, len(features)) for _ in range(10)]:
    plt.imshow(features[a], cmap="gray")
    plt.show()

# X_train, X_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.05, random_state=42)
#
# print("X_train.shape", X_train.shape)
# print("X_valid.shape", X_valid.shape)
# print("y_train.shape", y_train.shape)
# print("y_valid.shape", y_valid.shape)

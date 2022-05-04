from collections import Counter

import numpy as np


#////////////////////////////////////////

from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import LMTrP
import matplotlib.pyplot as plt

#.........................

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    #//////// process data ///////
    features = []
    labels = []
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images("./dataset_palm"))
    count = 0

    for (i, imagePath) in enumerate(imagePaths):
        j = 1
        count = count + 1
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        hist = LMTrP.LMTRP_process(image)
        # print(imagePath)
        # update the raw images, features, and labels matricies,
        # respectively
        features.append(hist)
        # print(features)
        # show an update every 1,000 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

        # if(i == 39):
        #     break

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    features = np.array(features)
    print(features.shape)


    for i in range(100):
        for j in range(10):
            labels.append(i)

    labels = np.array(labels)
    print(labels)

    print("[INFO] features matrix: {:.2f}MB".format(
        features.nbytes / (1024 * 1000.0)))

    nsamples, nx, ny = features.shape
    d2_train_dataset = features.reshape((nsamples,nx*ny))

    print(d2_train_dataset[0][0])

    #====================================

    X_train, X_test, y_train, y_test = train_test_split(
        d2_train_dataset, labels, test_size=0.2, random_state=1234
    )

    for k in range(1, 50):
        clf = KNN(k=k)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("KNN classification accuracy", accuracy(y_test, predictions))

        for test in y_test:
            print("KNN classification accuracy", accuracy(test, predictions))
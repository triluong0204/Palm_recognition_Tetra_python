# import the necessary packages
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

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "./dataset_palm", required=True,
# 	help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=1,
# 	help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# 	help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images("./dataset_palm"))
#imagePaths = "./dataset_palm/data"
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
features = []
labels = []

demo = []

# loop over the input images
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


# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    d2_train_dataset, labels, test_size=0.25, random_state=42)


#//////////////////////////////////
cmap = ListedColormap(['#FF0000', '#00FF00', "0000FF"])
plt.figure()
plt.scatter(d2_train_dataset[:, 0], d2_train_dataset[:, 1], c=labels, cmap=cmap, edgecolors='k', s=20)
plt.show()
#//////////////////////////////////


# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
# model = KNeighborsClassifier(n_neighbors=1, p = 2, weights = 'distance')
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

from sklearn.metrics import accuracy_score

error_rate = []

for i in range(1,50):
    print("k = ", i)
    model = KNeighborsClassifier(n_neighbors=i, p = 2, weights = 'uniform')
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)
    y_pred = model.predict(testFeat)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

import pickle


model1 = KNeighborsClassifier(n_neighbors=1, p = 2, weights = 'uniform')
model1.fit(trainFeat, trainLabels)
f = open("model.cpickle", "wb")
f.write(pickle.dumps(model1))
f.close()
print("----------------------------------------------")

for i in range(1,50):
    print("k = ", i)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(trainFeat, trainLabels)
    pred = knn.predict(testFeat)
    error_rate.append(np.mean(pred != testLabels))

print("Print results for 20 test data points:")
print("Predicted labels: ", pred[20:80])
print("Ground truth    : ", testLabels[20:80])

plt.figure(figsize=(15,10))
plt.plot(range(1,50),error_rate, marker='o', markersize=9)
plt.show()


cv2.waitKey(0)
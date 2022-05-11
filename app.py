import knn_ver3
import os
import cv2
import pickle
import LMTrP
import numpy as np

def predict(image, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    #image = cv2.imread(X_img_path)
    faces_encodings = LMTrP.LMTRP_process(image)

    faces_encodings = np.array(faces_encodings)
    nx, ny = faces_encodings.shape
    faces_encodings = faces_encodings.reshape(1, -1)

    print(knn_clf.predict_proba(faces_encodings))
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    return knn_clf.predict(faces_encodings)

if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    # print("Training KNN classifier...")
    # classifier = train("dataset_palm/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")

    image =  cv2.imread("0001_0001.bmp")

    predictions = predict(image, model_path="trained_knn_model.clf")
    print(predictions)

    # STEP 2: Using the trained classifier, make predictions for unknown images
    # for image_file in os.listdir("dataset_palm/test"): 
    #     #full_file_path = os.path.join("dataset_palm/test", image_file)
    #     full_file_path = "dataset_palm/test" + "/" + image_file
    #     print(full_file_path)

    #     print("Looking for faces in {}".format(image_file))

    #     # Find all people in the image using a trained classifier model
    #     # Note: You can pass in either a classifier file name or a classifier model instance
    #     predictions = predict(full_file_path, model_path="trained_knn_model.clf")

    #     print(predictions)
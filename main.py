# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from glob import glob
from scipy import spatial
from numpy import dot
from numpy.linalg import norm


def getFeatures(handShapeFeatureExtractor, type: str = "train"):
    data_path = os.getcwd()
    labels = []
    features = []
    video_paths = []
    if type == "train":
        data_path = os.path.join(data_path, "traindata")
        for file in os.listdir(data_path):
            if file.endswith(".mp4"):
                video_paths.append(os.path.join(data_path, file))
    else:
        data_path = os.path.join(data_path, "test")
        for file in os.listdir(data_path):
            if file.endswith(".mp4"):
                video_paths.append(os.path.join(data_path, file))
    frame_dir_path = os.path.join(data_path, "frames")
    for i, video_path in enumerate(video_paths):
        vp = os.path.join(data_path, video_path)
        frame = frameExtractor(vp, frame_dir_path, i)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features.append(np.squeeze(handShapeFeatureExtractor.extract_feature(frame)))
        if type == "train":
            labels.append((vp.split("/")[-1]).split("_")[0])
        else:
            labels.append(vp)
    return features, labels
    # return image_tensor

## import the handfeature extractor class
handShapeFeatureExtractor = HandShapeFeatureExtractor()
# =============================================================================
# Get the penultimate layer for trainig data
# your code goes here
# Extract the middle frame of each gesture video
training_features, training_labels = getFeatures(handShapeFeatureExtractor, "train")
test_features, test_labels = getFeatures(handShapeFeatureExtractor, "test")
print(training_labels)
results = []
for i, test_feature in enumerate(test_features):
    sim = []
    print("-----------------------------------------------------")
    for training_feature in training_features:
        # result = 1-spatial.distance.cosine(test_feature, training_feature)
        result = dot(test_feature, training_feature)/(norm(test_feature)*norm(training_feature))
        sim.append(result)
    # print(dist)
    # print(min(dist))

    ind = sim.index(max(sim))
    print(ind, test_labels[i], training_labels[ind])
    if training_labels[ind].startswith("Num"):
        results.append(int(training_labels[ind][-1]))
    elif training_labels[ind] == "FanOn":
        results.append(11)
    elif training_labels[ind] == "FanOff":
        results.append(12)
    elif training_labels[ind] == "LightOff":
        results.append(14)
    elif training_labels[ind] == "LightOn":
        results.append(15)
    elif training_labels[ind] == "SetThermo":
        results.append(16)
    elif training_labels[ind] == "FanUp":
        results.append(13)
    elif training_labels[ind] == "FanDown":
        results.append(10)

np.savetxt("Results.csv", results, delimiter=",", fmt='% d')
# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video




# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

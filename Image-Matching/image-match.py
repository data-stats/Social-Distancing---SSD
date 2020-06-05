import utils
from utils import LRN2D
import glob
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import load_model
from keras.engine.topology import Layer
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Defining Model
model = load_model('model.h5', custom_objects={'tf': tf})


def image_to_embedding(image, model):
    image = cv2.resize(image, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def recognize_face(face_image, input_embeddings, model):

    embedding = image_to_embedding(face_image, model)

    minimum_distance = 200
    name = None

    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():

        euclidean_distance = np.linalg.norm(embedding-input_embedding)

        print('Euclidean distance from %s is %s' %
              (input_name, euclidean_distance))

        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name

    if minimum_distance < 0.68:
        return str(name)
    else:
        return None


def create_input_image_embeddings():
    input_embeddings = {}

    for file in glob.glob("images/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(image_file, model)

    return input_embeddings


def recognize_faces_in_cam(input_embeddings):

    cv2.namedWindow("Face Recognizer")
    vc = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        height, width, channels = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through all the faces detected
        identities = []
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h

            face_image = frame[max(0, y1):min(
                height, y2), max(0, x1):min(width, x2)]
            identity = recognize_face(face_image, input_embeddings, model)

            if identity is not None:
                img = cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img, "MATCHED", (x1+5, y1-5),
                            font, 1, (255, 255, 255), 2)
            else:
                img = cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img, "NO MATCH FOUND", (x1+5, y1-5),
                            font, 1, (255, 255, 255), 2)

        key = cv2.waitKey(100)
        cv2.imshow("Face Matching", img)

        if key == 27:  # exit on ESC
            break
    vc.release()
    cv2.destroyAllWindows()


input_embeddings = create_input_image_embeddings()
recognize_faces_in_cam(input_embeddings)

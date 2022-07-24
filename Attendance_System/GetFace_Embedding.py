from imutils import paths
import numpy
import imutils
import pickle
import cv2
import os
import time

"""
https://www.educba.com/python-pickle-vs-json/
https://www.geeksforgeeks.org/facenet-using-facial-recognition-system/#:~:text=FaceNet%20is%20the%20name%20of,for%20Face%20Recognition%20and%20Clustering.

FaceNet takes an image of the person's face as input and outputs a vector of 128 numbers
which represent the most important features of a face. In machine learning, this vector
is called embedding.

OpenFace is an open-source toolkit based on the FaceNet algorithm for automatic 
facial identification that was created by Google.OpenFace is an open source tool
intended for computer vision and machine learning researchers, the affective
computing community and people interested in building interactive applications
based on facial behavior analysis.
"""

embeddings_file_name = "embeddings/face_embeddings.pickle"

model_path = "caffe_FDM/res10_300x300_ssd_iter_140000.caffemodel"
model_proto_txt = "caffe_FDM/deploy.prototxt"
embedding_model_path = "Openface_Facenet_model/openface_nn4.small2.v1.t7"


face_detect_model = cv2.dnn.readNetFromCaffe(model_proto_txt,
                                             model_path)
print("Caffe face detection model loaded...")

embedded_model = cv2.dnn.readNetFromTorch(embedding_model_path)
print("OpenFace FaceNet model loaded")

image_paths = paths.list_images("dataset")
embeddings = []
names = []

confidence_thresh_hold = 0.5

for i, path in enumerate(image_paths):
    name = path.split(os.path.sep)[-2]
    image1 = cv2.imread(path)
    image2 = imutils.resize(image1,
                            width=600)
    h, w = image2.shape[:2]

    blob_image = cv2.dnn.blobFromImage(cv2.resize(image2, (300, 300)),
                                       1.0, (300, 300),
                                       (104.0, 177.0, 123.0),
                                       swapRB=False,
                                       crop=False)

    face_detect_model.setInput(blob_image)

    detections = face_detect_model.forward()

    if len(detections) > 0:
        i = numpy.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_thresh_hold:
            box = detections[0, 0, i, 3:7]*numpy.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")

            face = image2[start_y: end_y, start_x: end_x]

            #  cv2.imshow("model detected blob", face)
            #  time.sleep(5)

            fh, fw = face.shape[:2]
            print("Face detected")
            if fw < 20 or fh < 20:
                print("Face constraint met, scanning next image...")
                continue

            blob_face = cv2.dnn.blobFromImage(face,
                                              1.0/255,
                                              (96, 96),
                                              (0, 0, 0),
                                              swapRB=True,
                                              crop=False)

            embedded_model.setInput(blob_face)
            embedding_vector = embedded_model.forward()
            print("Embedding found...")
            names.append(name)
            embeddings.append(embedding_vector.flatten())


data = {"embeddings": embeddings, "names": names}
#  print(embeddings[-1]) [ 4.89967735e-03  2.89688744e-02  .. .. .]
#  print(len(embeddings[-1])) output = 128

with open(embeddings_file_name, "wb") as file:
    file.write(pickle.dumps(data))
    print("Embedding file saved.")




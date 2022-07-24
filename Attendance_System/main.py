"""
real time recognition
"""

import numpy
import imutils
import pickle
import cv2

embedding_model_path = "Openface_Facenet_model/openface_nn4.small2.v1.t7"

model_path = "caffe_FDM/res10_300x300_ssd_iter_140000.caffemodel"
model_proto_txt = "caffe_FDM/deploy.prototxt"

svc_model_file = "svm_model/svm_model.pickle"
label_encoder_file = "svm_model/label_encoding.pickle"

face_detect_model = cv2.dnn.readNetFromCaffe(model_proto_txt,
                                             model_path)
print("Caffe face detection model loaded...")

embedded_model = cv2.dnn.readNetFromTorch(embedding_model_path)
print("OpenFace FaceNet model loaded")

with open(svc_model_file, 'rb') as file:
    reader = file.read()
    svc_classifier = pickle.loads(reader)
print("svc classifier loaded")

with open(label_encoder_file, 'rb') as file:
    reader = file.read()
    label_encoder = pickle.loads(reader)
print("Label encoder loaded")

video = cv2.VideoCapture(0)
confidence_thresh_hold = 0.5
while True:
    flag, live = video.read()
    if flag:
        live = imutils.resize(live, width=600)
        h, w = live.shape[:2]

        blob_image = cv2.dnn.blobFromImage(cv2.resize(live, (300, 300)),
                                           1.0,
                                           (300, 300),
                                           (104.0, 177.0, 123.0),
                                           swapRB=False,
                                           crop=False)

        face_detect_model.setInput(blob_image)
        detections = face_detect_model.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_thresh_hold:
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype("int")

                face = live[start_y: end_y, start_x: end_x]
                fh, fw = face.shape[:2]

                print("Face detected")
                if fw < 20 or fh < 20:
                    print("Face constraint met, scanning next image...")
                    continue

                blob_face = cv2.dnn.blobFromImage(face,
                                                  1.0 / 255,
                                                  (96, 96),
                                                  (0, 0, 0),
                                                  swapRB=True,
                                                  crop=False)

                embedded_model.setInput(blob_face)
                embedding_vector = embedded_model.forward()
                print("Embedding found...")

                prediction = svc_classifier.predict_proba(embedding_vector)[0]
                j = numpy.argmax(prediction)
                proba = prediction[j]

                name = label_encoder[j]

                text = f"{name}, p={proba * 100:.2f}"

                cv2.rectangle(live,
                              (start_x, start_y),
                              (end_x, end_y),
                              (0, 255, 0),
                              2)

                start_y = start_y - 10 if start_y - 10 > 10 else start_y + 10

                cv2.putText(live,
                            text,
                            (start_x, start_y),
                            cv2.FONT_HERSHEY_PLAIN,
                            0.45,
                            (0, 255, 0),
                            2)

        cv2.imshow("Attendance System", live)
        key_press = cv2.waitKey(1)
        if key_press == 32:  # space bar
            break

video.release()
cv2.destroyAllWindows()

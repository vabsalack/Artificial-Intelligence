import numpy
import imutils
import cv2
import numpy as np


def object_recognition():

    proto_txt = "MobileNetSSD_deploy.prototxt.txt"
    model_file = "MobileNetSSD_deploy.caffemodel"
    conf_thresh = 0.2

    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train",
               "tv monitor"]

    colors = numpy.random.uniform(0, 255, size=(len(classes), 3))

    print(">Loading model...")
    model = cv2.dnn.readNetFromCaffe(proto_txt, model_file)
    print(">Model loaded...")

    video = cv2.VideoCapture(0)

    while True:
        flag, live = video.read()
        if flag:
            live = imutils.resize(live, width=500)
            (h, w) = live.shape[:2]
            live_resize = cv2.resize(live, (300, 300))
            blob_live = cv2.dnn.blobFromImage(live_resize, 0.007843, (300, 300), 127.5)

            model.setInput(blob_live)
            detections = model.forward()
            detection_shape = detections.shape[2]

            for i in np.arange(0, detection_shape):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_thresh:
                    idx = int(detections[0, 0, i, 1])
                    print("classes id: ", detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7]*numpy.array([w, h, w, h])
                    (startx, starty, endx, endy) = box.astype("int")

                    label = f"{classes[idx]}: {confidence*100:.2f}"

                    cv2.rectangle(live, (startx, starty), (endx, endy), colors[idx], 2)
                    if starty - 15 > 15:
                        y = starty - 15
                    else:
                        y = starty + 15
                    cv2.putText(live, label, (startx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

            cv2.imshow("live", live)
            key_press = cv2.waitKey(1)
            if key_press == 32:
                break

    video.release()
    cv2.destroyAllWindows()

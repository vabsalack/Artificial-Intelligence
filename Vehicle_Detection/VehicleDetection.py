import cv2
import imutils


def car_detection():

    cascade_file = "haarcascade_cars.xml"
    classifier = cv2.CascadeClassifier(cascade_file)

    video = cv2.VideoCapture(0)

    while True:
        flag, live = video.read()
        if flag:
            image = imutils.resize(live, width=300)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cars = classifier.detectMultiScale(gray_image,
                                               1.1,
                                               1)

            for x, y, w, h in cars:
                cv2.rectangle(live, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Car Detection", live)

            vehicle_detected = len(cars)
            print(f"{vehicle_detected} vehicle detected")

        key_press = cv2.waitKey(1)
        if key_press == 32:
            break

    video.release()
    cv2.destroyAllWindows()














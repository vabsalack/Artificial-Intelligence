from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound

"""
    install dlib library compatible with python 3.10.1
    <pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl>
"""

lm_dat_file = "shape_predictor_68_face_landmarks.dat"


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ratio = (A + B)/(2.0 * C)
    return ratio


def drowsiness_detector():

    ratio_thresh_hold = 0.3
    maximum_frames = 24
    count = 0

    frequency = 2500
    duration = 1000

    video = cv2.VideoCapture(0)
    face_detector = dlib.get_frontal_face_detector()
    marker = dlib.shape_predictor(lm_dat_file)

    #  left and right eye coordinates
    left_eye_start, left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start, right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        flag, live = video.read()
        if flag:
            live = imutils.resize(live, width=450)
            gray_live = cv2.cvtColor(live, 6)

            faces = face_detector(gray_live)

            for face in faces:
                landmarks = marker(gray_live, face)
                landmarks = face_utils.shape_to_np(landmarks)

                left_eye = landmarks[left_eye_start:left_eye_end]
                right_eye = landmarks[right_eye_start:right_eye_end]

                left_aspect_ratio = eye_aspect_ratio(left_eye)
                right_aspect_ratio = eye_aspect_ratio(right_eye)

                average_aspect_ratio = (left_aspect_ratio + right_aspect_ratio) / 2.0

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)

                cv2.drawContours(live,
                                 left_eye_hull,
                                 -1,
                                 (0, 255, 0),
                                 2)
                cv2.drawContours(live,
                                 right_eye_hull,
                                 -1,
                                 (0, 255, 0),
                                 2)

                if average_aspect_ratio < ratio_thresh_hold:
                    count += 1

                    if count > maximum_frames - 1:
                        cv2.putText(live,
                                    "DROWSINESS DETECTED",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    0.7,
                                    (0, 255, 0),
                                    2)

                        winsound.Beep(frequency,
                                      duration)

                else:
                    count = 0

            cv2.imshow("Recording", live)
            key_press = cv2.waitKey(1)

            if key_press == 32:
                break

    video.release()
    cv2.destroyAllWindows()



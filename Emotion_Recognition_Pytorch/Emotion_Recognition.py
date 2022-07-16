from facial_emotion_recognition import EmotionRecognition
import cv2


def Emotion_Recognizer():
    emotion_model = EmotionRecognition(device="cpu")
    video = cv2.VideoCapture(0)

    while True:

        flag, live = video.read()

        if flag:

            image = emotion_model.recognise_emotion(live, return_type="BGR")
            cv2.imshow("Emotion Recognizer", image)
            key_press = cv2.waitKey(1)

            if key_press == 32:
                break

    video.release()
    cv2.destroyAllWindows()

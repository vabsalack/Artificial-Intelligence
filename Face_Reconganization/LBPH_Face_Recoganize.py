import cv2
import numpy
import os


def preprocess_training_data(folder="saved_faces"):

    images, labels, names, idd = list(), list(), dict(), int()

    folder_name = folder
    for root, dir_names, file_names in os.walk(folder_name):

        for sub_dir in dir_names:

            names[idd] = sub_dir
            subject_path = os.path.join(root, sub_dir)

            for image_name in os.listdir(subject_path):

                path = os.path.join(subject_path, image_name)
                images.append(cv2.imread(path, 0))
                labels.append(int(idd))

            idd += 1

    images, labels = [numpy.array(i) for i in (images, labels)]

    return images, labels, names


def model_train(images, labels):
    model_name = [cv2.face.LBPHFaceRecognizer_create, cv2.face.FisherFaceRecognizer_create]
    print(">LBPH_FaceRecognizer: 0", ">Fisher_FaceRecognizer: 1", sep="\n")
    model_no = int(input(">Enter Recognizer no: "))
    model = model_name[model_no]()
    model.train(images, labels)
    return model


def Face_Recognition(model, names):

    haar = "haarcascade_frontalface_alt.xml"
    classifier = cv2.CascadeClassifier(haar)

    video = cv2.VideoCapture(0)

    while True:

        flag, live = video.read()

        if flag:
            gray_image = cv2.cvtColor(live, 6)
            faces = classifier.detectMultiScale(gray_image, 1.3, 5)
            for x, y, w, h in faces:
                cv2.rectangle(live, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cropped_face = gray_image[y:y+h, x:x+w]
                cropped_face = cv2.resize(cropped_face, (200, 200))

                prediction = model.predict(cropped_face)
                text_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

                if prediction[1] <= 800:

                    person_name = f"{names[prediction[0]]}"

                    (wt, ht), _ = cv2.getTextSize(person_name.upper(), text_font, 1, 1)

                    cv2.rectangle(live, (x, y-ht), (x+wt, y), (0, 255, 0), -1)
                    cv2.putText(live, person_name.upper(), (x, y), text_font, 1, (255, 255, 255), 1)
                else:
                    cv2.putText(live, "stranger !!!", (x, y), text_font, 1, (255, 255, 255), 1)

            cv2.imshow("FARE", live)

            if cv2.waitKey(1) == 32:
                break

    video.release()
    cv2.destroyAllWindows()





import cv2
import os


def create_image_folder(folder, image_folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    path = os.path.join(folder, image_folder)

    if not os.path.isdir(path):
        os.mkdir(path)


def Frontal_Face_Detect():
    haar = "haarcascade_frontalface_alt.xml"
    classifier = cv2.CascadeClassifier(haar)

    video = cv2.VideoCapture(0)  # system primary camera

    while True:
        flag, live = video.read()
        if flag:

            gray_image = cv2.cvtColor(live, 6)  # module attribute cv2.COLOR_BGR2GRAY = 6

            faces = classifier.detectMultiScale(gray_image, 1.1, 5)

            for x, y, w, h in faces:
                cv2.rectangle(live, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Face Detection", live)

            if cv2.waitKey(1) == 32:
                break

    video.release()
    cv2.destroyAllWindows()


def create_image_dataset():
    folder_name = input(">enter folder name: ")
    image_name = input(">enter your name: ")
    image_count = int(input(">enter no of images: "))
    print("press 's' to save image..", "press 'space' to quit", sep="\n")

    create_image_folder(folder_name, image_name)

    haar = "haarcascade_frontalface_alt.xml"
    classifier = cv2.CascadeClassifier(haar)

    video = cv2.VideoCapture(0)  # system primary camera

    images_saved = 0

    while images_saved < image_count:
        flag, live = video.read()
        if flag:

            gray_image = cv2.cvtColor(live, 6)  # module attribute cv2.COLOR_BGR2GRAY = 6

            faces = classifier.detectMultiScale(gray_image, 1.1, 5)

            for x, y, w, h in faces:
                cv2.rectangle(live, (x, y), (x + w, y + h), (0, 255, 0), 2)
                gray_face = gray_image[y:y + h, x:x + w]
                resized_face = cv2.resize(gray_face, (200, 200))

            cv2.imshow("Face Detection", live)

            key_press = cv2.waitKey(1)

            if key_press == 115:
                try:
                    cv2.imwrite(f"{folder_name}//{image_name}//{image_name}_{images_saved + 1}.jpg", resized_face)
                    print(f"{image_name}_{images_saved + 1}.jpg saved...")
                    images_saved += 1
                except:
                    print("Image saving failed", "Face isn't detected...", end="\n")

            elif key_press == 32:
                break

    video.release()
    cv2.destroyAllWindows()

import cv2
import os
import csv


def create_image_folder(folder, image_folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    path = os.path.join(folder, image_folder)

    if not os.path.isdir(path):
        os.mkdir(path)


def update_csv(file, info):
    with open(file, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(info)


def create_image_dataset_1():
    image_name = input(">enter your name: ")
    roll_no = int(input(">enter your roll no. : "))
    update_csv("students.csv", [image_name, roll_no])
    image_count = int(input(">enter no of images: "))
    print("press 's' to save image..", "press 'space' to quit", sep="\n")

    create_image_folder("dataset", image_name)

    haar = "haarcascade_frontalface_alt.xml"
    classifier = cv2.CascadeClassifier(haar)

    video = cv2.VideoCapture(0)  # system primary camera

    images_saved = 0

    while images_saved < image_count:
        flag, live = video.read()
        if flag:

            gray_image = cv2.cvtColor(live, 6)  # module attribute cv2.COLOR_BGR2GRAY = 6

            faces = classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(30, 30))
            live_01 = live.copy()

            for x, y, w, h in faces:
                cv2.rectangle(live_01, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Face Detection", live_01)

            key_press = cv2.waitKey(1)

            if key_press == 115:
                try:
                    cv2.imwrite(f"dataset//{image_name}//{image_name}_{images_saved + 1}.jpg", live)
                    print(f"{image_name}_{images_saved + 1}.jpg saved...")
                    images_saved += 1
                except:
                    print("Image saving failed", "Face isn't detected...", end="\n")

            elif key_press == 32:
                break

    video.release()
    cv2.destroyAllWindows()


def create_image_dataset():

    image_name = input(">enter your name: ")
    roll_no = int(input(">enter your roll no. : "))
    update_csv("students.csv", [image_name, roll_no])
    image_count = int(input(">enter no of images: "))
    print("press 's' to save image..", "press 'space' to quit", sep="\n")

    create_image_folder("dataset", image_name)

    haar = "haarcascade_frontalface_alt.xml"
    classifier = cv2.CascadeClassifier(haar)

    video = cv2.VideoCapture(0)  # system primary camera

    images_saved = 0

    while images_saved < image_count:
        flag, live = video.read()
        if flag:

            gray_image = cv2.cvtColor(live, 6)  # module attribute cv2.COLOR_BGR2GRAY = 6

            faces = classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(30, 30))

            for x, y, w, h in faces:
                cv2.rectangle(live, (x, y), (x + w, y + h), (0, 255, 0), 2)
                gray_face = gray_image[y:y + h, x:x + w]
                resized_face = cv2.resize(gray_face, (200, 200))

            cv2.imshow("Face Detection", live)

            key_press = cv2.waitKey(1)

            if key_press == 115:
                try:
                    cv2.imwrite(f"dataset//{image_name}//{image_name}_{images_saved + 1}.jpg", resized_face)
                    print(f"{image_name}_{images_saved + 1}.jpg saved...")
                    images_saved += 1
                except:
                    print("Image saving failed", "Face isn't detected...", end="\n")

            elif key_press == 32:
                break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    create_image_dataset_1()
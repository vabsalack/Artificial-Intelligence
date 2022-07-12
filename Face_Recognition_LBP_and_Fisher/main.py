from LBPH_Face_Recoganize import *
from FaceDetect import *


def main():
    action = input("Create Dataset [yes/no]: ").lower()
    if action == "yes":

        print(">creating dataset...")
        create_image_dataset()
        print(">dataset creation completed")

    print(">preprocessing dataset...")
    images, labels, names = preprocess_training_data()
    print(">preprocessing completed")

    print(">model training started...")
    model = model_train(images, labels)
    print(">model training completed")

    print(">Recognizing faces...")
    Face_Recognition(model, names)
    print(">Program terminated")


if __name__ == "__main__":
    main()
from download_image import *
from convolution_model import *
from renaming_image import *


def image_creation():
    image_name = input(">enter image name: ")
    count = int(input(">enter number of image: "))
    download_image(image_name, count)


def main2():

    model = create_model()
    training_set, validation_set = data_set("train", "validate")
    model = train_model(model, training_set, validation_set)
    save_model(model, "model01")


def main1():
    image_creation()
    image_creation()

    rename_files(input(">enter folder name: "))
    rename_files(input(">enter folder name: "))


if __name__ == "__main__":
    function = [main1, main2]
    action = int(input("did you created datasets of two unique images [YES(1)/NO(0)] press 0 or 1> "))
    if action == 1:
        print("training model...")
        function[1]()
    else:
        print("creating dataset...")
        function[0]()




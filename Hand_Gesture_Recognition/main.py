from Gesture_Model import *
from test_model import *


def main_train():

    print("creating CNN model...")
    model = create_model()

    training_set, validating_set = data_feed("train", "validate")

    print("model created \n training started...")
    train_model(model, training_set, validating_set)
    print("training completed. \n model get saved automatically model_name='model.h5' 'save_best_only=True'...")


def main_test():

    """ returns generator, which next() to classify images in test folder"""
    model = model_load("model.h5")
    image_files = list_images("dataset//test")

    return classify(model, image_files)


if __name__ == "__main__":
    function = [main_train, main_test]
    action = int(input(">is model trained and saved in disk? yes(1)/no(0): "))
    if action:
        iterator = function[action]()
        next(iterator)
        next(iterator)
        next(iterator)
        next(iterator)
        next(iterator)
        next(iterator)
        next(iterator)
        next(iterator)
        next(iterator)
        next(iterator)
    else:
        function[action]()



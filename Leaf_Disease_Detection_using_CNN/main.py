from LeafCNNModel import *
from test import *


def main_train():

    model = create_model()

    training_set, validating_set = data_feed("train", "validate")
    model1 = train_model(model, training_set, validating_set)

    save_model(model1)


def main_test():

    model = load_model("model.json", "model_w.h5")
    images = list_images("test")
    classify(model, images)


if __name__ == "__main__":

    function = [main_train, main_test]
    action = int(input(">is model trained and saved in disk? yes(1)/no(0): "))
    if action:
        function[action]()
    else:
        function[action]()
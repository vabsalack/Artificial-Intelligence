from KERAS_MODEL import *


def create_train_model():
    #  creation of model
    model = create_model()

    #  loading dataset, inputs and target
    inputs, outputs = load_dataset("pima-indians-diabetes.csv")

    #  train model
    model_train = train_model(model, inputs, outputs)

    #  saving model and weights
    save_model(model_train, "model01")


def test_model():

    model = load_model("model01.json")
    model_predict(model, "pima-indians-diabetes.csv")


if __name__ == "__main__":
    # create_train_model()
    test_model()
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from random import randint


def load_dataset(file_location):

    print(">Loading dataset...")
    dataset = loadtxt(file_location, delimiter=",")  # return 2D numpy array of elements of row
    inputs = dataset[:, 0:8]
    outputs = dataset[:, 8]

    return inputs, outputs


def create_model():

    print(">Creating model...")
    model = Sequential()
    model.add(Dense(8, activation="relu"))
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(12, input_dim=12, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def train_model(model, inputs, outputs):

    print(">Training model...")
    model.fit(inputs, outputs, epochs=5, batch_size=10)
    _, accuracy = model.evaluate(inputs, outputs)
    print(f"Accuracy: {accuracy*100:.2f}")
    return model


def save_model(model, file_name):

    print(">Saving model...")
    model_json = model.to_json()
    with open(f"{file_name}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Model Saved to disk...")


def load_model(file_name):

    print(">Loading model...")
    with open(file_name, "r") as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    print(">Loaded model from disk...")
    return model


def model_predict(model, data_location):

    x, y = load_dataset(data_location)
    predictions = model.predict_step(x)

    for _ in range(10):
        i = randint(0, 767)
        print(f"{x[i].tolist()} => {predictions[i]} (expected {y[i]})")















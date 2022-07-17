from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


def create_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    return model


def data_set(train, validate):

    train_data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    validate_data_generator = ImageDataGenerator(rescale=1./255)

    training_set = train_data_generator.flow_from_directory(f"simple_images//{train}", target_size=(64, 64), batch_size=8, class_mode="binary")
    validation_set = validate_data_generator.flow_from_directory(f"simple_images//{validate}", target_size=(64, 64), batch_size=8, class_mode="binary")

    return training_set, validation_set


def train_model(model, train_data, validate_data):
    model.fit_generator(train_data, steps_per_epoch=2, epochs=10, validation_data=validate_data, validation_steps=2)
    return model


def save_model(model, model_name):
    print(">saving model...")
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("model saved to disk")









from keras.models import Sequential

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator


def create_model():

    #  global model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(25, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def data_feed(train, validate):

    train_set_generator = ImageDataGenerator(rescale=None,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)

    validate_set_generator = ImageDataGenerator(rescale=1. / 255)

    training_set = train_set_generator.flow_from_directory(f"dataset/{train}",
                                                           target_size=(128, 128),
                                                           batch_size=10,
                                                           class_mode="categorical")

    print(training_set.class_indices)

    validating_set = validate_set_generator.flow_from_directory(f"dataset/{validate}",
                                                                target_size=(128, 128),
                                                                batch_size=10,
                                                                class_mode="categorical")
    print(validating_set.class_indices)

    return training_set, validating_set


def train_model(model, training_set, validating_set):

    model.fit_generator(training_set,
                        steps_per_epoch=20,
                        epochs=25,
                        validation_data=validating_set,
                        validation_steps=10)

    return model


def save_model(model):

    print("Saving model...")

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model_w.h5')

    print("Model Saved Successful.")






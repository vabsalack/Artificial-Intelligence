from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=150, activation="relu"))
    model.add(Dropout(0.25))

    model.add(Dense(units=6, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def data_feed(train, validate):
    train_set_generator = ImageDataGenerator(rescale=1. / 255, rotation_range=12., width_shift_range=0.2,
                                             height_shift_range=0.2, zoom_range=0.15, horizontal_flip=True)
    validate_set_generator = ImageDataGenerator(rescale=1. / 255)

    training_set = train_set_generator.flow_from_directory(f"dataset/{train}", target_size=(256, 256),
                                                           color_mode="grayscale",
                                                           batch_size=8, classes=["0_NONE", "1_ONE", "2_TWO", "3_THREE", "4_FOUR",
                                                                                  "5_FIVE"], class_mode="categorical")

    validating_set = validate_set_generator.flow_from_directory(f"dataset/{validate}", target_size=(256, 256),
                                                                color_mode="grayscale", batch_size=8,
                                                                classes=["0_NONE", "1_ONE", "2_TWO", "3_THREE", "4_FOUR", "5_FIVE"],
                                                                class_mode="categorical")

    return training_set, validating_set


def train_model(model, training_set, validating_set):
    call_backs = [EarlyStopping(monitor="val_loss", patience=5),
                  ModelCheckpoint(filepath="model.h5", monitor="val_loss", save_best_only=True, verbose=1)]

    model.fit_generator(training_set, steps_per_epoch=20, epochs=50, validation_data=validating_set, validation_steps=7,
                        callbacks=call_backs)




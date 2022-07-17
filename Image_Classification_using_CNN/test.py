from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image


def load_model(model_name):

    print(">loading model from disk...")
    with open(model_name, "r") as json_file:
        json_model = json_file.read()
    model = model_from_json(json_model)
    model.load_weights("model.h5")
    print(">model loaded from disk.")

    return model


def classify(model, image_file):

    for img in image_file:
        image_name = img
        test_image = image.image_utils.load_img(image_name, target_size=(64, 64))
        test_image = image.image_utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = "car"
        else:
            prediction = "bike"

        yield prediction, image_name



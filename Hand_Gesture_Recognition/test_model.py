from keras.preprocessing import image
from keras.models import load_model
import numpy, os


def model_load(model_name):
    print(">loading model...")
    model = load_model(model_name)
    print("model loaded successfully.")
    return model


def list_images(root_folder):

    files = []
    for root, sub_dir, file in os.walk(root_folder):
        for f in file:
            if ".png" in f:
                files.append(os.path.join(root, f))

    return files


def classify(model, image_files):

    for image_path in image_files:

        test_image = image.image_utils.load_img(image_path, target_size=(256, 256), grayscale=True)

        test_image = image.image_utils.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)

        result = model.predict(test_image)
        arr = numpy.array(result[0])

        print(arr)
        max_prob = arr.argmax(axis=0)

        classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
        result = classes[max_prob]

        print(image_path, result)
        yield











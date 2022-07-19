import numpy
from keras.preprocessing import image
from keras.models import model_from_json
import os


def load_model(model_name, model_weights):

    print("Loading model...")
    json_file = open(f"{model_name}", "r")
    json_model = json_file.read()
    json_file.close()

    model = model_from_json(json_model)
    model.load_weights(model_weights)

    print("Model Loaded Successfully.")

    return model


def list_images(test):

    files = []
    for root, sub_dir, file in os.walk(f"dataset//{test}"):
        for f in file:
            if ".JPG" in f:
                files.append(os.path.join(root, f))

    return files


def classify(model, files):

    label = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
             "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
             "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
             "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
             "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
             "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
             "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
             "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"]

    for file in files:
        test_image = image.image_utils.load_img(file, target_size=(128, 128))
        test_image = image.image_utils.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)

        result = model.predict(test_image)
        #  print("result:", result)

        arr = numpy.array(result)
        #  print(arr)
        print(arr.argmax())
        print(f"image: {file} || result: {label[arr.argmax()]}")



from test import *
import os


def images_paths(root_folder):
    images_path = []

    for root, dir_names, files in os.walk(root_folder):
        for file in files:
            if ".jpeg" in file:
                images_path.append(os.path.join(root, file))

    return images_path


def main():
    model = load_model("model01.json")
    images = images_paths("simple_images")

    iterator = classify(model, images)

    print(*next(iterator))
    print(*next(iterator))
    print(*next(iterator))
    print(*next(iterator))
    print(*next(iterator))
    print(*next(iterator))


if __name__ == "__main__":
    main()
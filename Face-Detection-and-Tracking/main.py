from FaceDetect import *


def main():
    functions = [Frontal_Face_Detect, create_image_dataset]

    print("press [0] Zero for Face Detection", "press [1] one for saving Detected Face", sep="\n")

    action = int(input("input: "))

    inputs = [0, 1]
    if action in inputs:
        functions[action]()
    else:
        print("invalid input")


if __name__ == "__main__":
    main()
import Emotion_Recognition


def main():
    action = input(">Open Emotion Recognizer? [YES/NO]: ")
    if action.lower() == "yes":
        print("running Emotion Recognizer...")
        Emotion_Recognition.Emotion_Recognizer()
        print("program terminated from user end")
    else:
        print("Quiting process...")


if __name__ == "__main__":
    main()
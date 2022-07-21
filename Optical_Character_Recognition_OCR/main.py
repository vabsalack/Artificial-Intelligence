from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def image_to_string(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image, lang="eng")
    return text


def main():
    files = ["image1.png", "image3.png"]
    text = [image_to_string(file) for file in files]
    print(*text, sep="\n")


if __name__ == "__main__":
    main()





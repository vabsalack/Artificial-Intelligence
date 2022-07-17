import os


def rename_files(file_image):
    os.chdir(f"C:\\non - window\\REMOTE GIT\\Artificial-Intelligence\\Image_Classification_using_CNN\\simple_images\\{file_image}")
    i = 1
    for file in os.listdir():
        source = file
        destination = f"{file_image}_" + str(i) + ".jpeg"
        os.rename(source, destination)
        i += 1




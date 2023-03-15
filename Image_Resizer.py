from PIL import Image
import os

def webpToJpeg(string):
    im = Image.open(string).convert("RGB")
    im.save(string[:-4] + "jpg", "jpeg")

#string = "ExamplePics/0up1cmenb8ma1.webp"
#webpToJpeg(string)

def resize(image, size):
    resized_image = image.resize((size, size))
    return resized_image

#im = Image.open("ExamplePics/0up1cmenb8ma1.jpg")
#resize(im).save("ExamplePics/0up1cmenb8ma1_resized.jpg")
def resize_all(size):
    directory = "images"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        im = Image.open(f)
        resize(im, size).save(os.path.join("resized_images", filename))
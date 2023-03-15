from PIL import Image

def webpToJpeg(string):
    im = Image.open(string).convert("RGB")
    im.save(string[:-4] + "jpg", "jpeg")

string = "ExamplePics/0up1cmenb8ma1.webp"
webpToJpeg(string)

def resize(image):
    size = 800
    resized_image = image.resize((size, size))
    return resized_image

im = Image.open("ExamplePics/0up1cmenb8ma1.jpg")
resize(im).save("ExamplePics/0up1cmenb8ma1_resized.jpg")
from PIL import Image
import os
from numba import jit, cuda
import time
from datetime import timedelta
@jit(target_backend='cuda') 
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
    startTime=time.time()
    directory = "images"
    resize_directory = "resized_images"
    CHECK_FOLDER = os.path.isdir(resize_directory)
    if not CHECK_FOLDER:
        os.makedirs(resize_directory)

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        im = Image.open(f)
        resize(im, size).save(os.path.join(resize_directory, filename))
        os.remove(f)
    endTime=int(time.time()-startTime)
    td=timedelta(seconds=endTime)
    print("Time elapsed in hh:mm:ss | "+str(td))


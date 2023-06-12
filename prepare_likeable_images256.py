import numpy as np
import pandas as pd
from PIL import Image
import os

data = pd.read_csv('submissions.csv')
upvotes = data["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
data["Log_Upvotes"]=logupvotes

labels=np.quantile(data["Log_Upvotes"],0.9)

most_likeable_data = data[data["Log_Upvotes"] > labels]
size = 256
directory = "resized_images"
resize_directory = "least_likeable_data256"
CHECK_FOLDER = os.path.isdir(resize_directory)
if not CHECK_FOLDER:
    os.makedirs(resize_directory)


for ID in most_likeable_data['ID']:
    filename = "EarthPorn-" + ID + ".png"
    f = os.path.join(directory, filename)
    im = Image.open(f)
    im.resize((size, size)).save(os.path.join(resize_directory, filename[:-3] + "jpg"))

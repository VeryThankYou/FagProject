import numpy as np
import pandas as pd
from PIL import Image
import os
os.chdir(os.getcwd()+"/Data")
data = pd.read_csv('submissions.csv')
upvotes = data["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
data["Log_Upvotes"]=logupvotes



most = False

if most:
    labels=np.quantile(data["Log_Upvotes"],0.9)
    data_10_procent = data[data["Log_Upvotes"] > labels]
    resize_directory = "/most_likeable_data256"
else:
    labels=np.quantile(data["Log_Upvotes"],0.1)
    data_10_procent = data[data["Log_Upvotes"] < labels]
    resize_directory = "/least_likeable_data256"
size = 256
directory = os.getcwd() + "/resized_images256"
resize_directory_path = os.getcwd() + resize_directory
CHECK_FOLDER = os.path.isdir(resize_directory_path)
if not CHECK_FOLDER:
    os.makedirs(resize_directory_path)


for ID in data_10_procent['ID']:
    filename = "EarthPorn-" + ID + ".jpg"
    f = os.path.join(directory, filename)
    im = Image.open(f)
    im.resize((size, size)).save(os.path.join(resize_directory_path, filename))

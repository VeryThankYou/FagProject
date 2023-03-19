import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

os.chdir('/Volumes/Seagate Expansion Drive/Clara/DTU/Fagprojekt')


df = pd.read_csv("submissions.csv")
upvotes = df["Score"]

data_description = {}

data_description["Min"] = np.min(upvotes)
data_description["Max"] = np.max(upvotes)
data_description["Mean"] = round(np.mean(upvotes),1)
data_description["Std"] = round(np.std(upvotes),1)
data_description["Mode"] = np.bincount(upvotes).argmax()
data_description["Q_1"] = np.quantile(upvotes, q=0.25)
data_description["Median"] = np.median(upvotes)
data_description["Q_3"] = np.quantile(upvotes, q=0.75)


print(data_description)

#plt.hist(upvotes, bins = 20) #Histogram of distribution of upvotes
#plt.title("Upvotes")
#plt.show()

data_description = pd.DataFrame(data_description, index=[0])
print(data_description.style.to_latex(position = "H", position_float="centering", hrules = True))


# Average image
os.chdir('/Volumes/Seagate Expansion Drive/Clara/DTU/Fagprojekt/resized_images')
gray_scaled = True ### Gray-scaled or in colours

images_arrays = []

for id in df["ID"][0:10000]:
    image = Image.open('EarthPorn-' + id + '.png')
    if gray_scaled:
        image = np.asarray(image.convert('L'))
    else:
        image = np.asarray(image)
    images_arrays.append(image)


average_image = np.mean(images_arrays, axis = 0)

if gray_scaled:
    plt.imshow(average_image.astype('uint8'), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(average_image.astype('uint8'))
plt.show()


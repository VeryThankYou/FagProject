import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

# Import Data
os.chdir(os.getcwd()+"/Data")
df = pd.read_csv("submissions.csv")
upvotes = df["Score"].to_numpy()

# Data Description
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

# Histograms

os.chdir(os.path.dirname(os.getcwd())+'/Data_Analysis')
plt.hist(upvotes, bins = 20) #Histogram of distribution of upvotes
plt.title("Upvotes")
plt.savefig("Histogram_upvotes.png")
plt.show()


logupvotes = np.log(upvotes+1)
print(len(logupvotes))
plt.hist(logupvotes)
plt.savefig("Histogram_log_upvotes.png")
plt.show()


data_description = pd.DataFrame(data_description, index=[0])
print(data_description.style.to_latex(position = "H", position_float="centering", hrules = True))


# Average image
os.chdir(os.path.dirname(os.getcwd())+"/Data/resized_images256")
gray_scaled = False ### Gray-scaled or in colours

images_arrays = []
amount = 1000
for id in df["ID"][0:amount]:
    image = Image.open('EarthPorn-' + id + '.jpg')
    if gray_scaled:
        image = np.asarray(image.convert('L'))
    else:
        image = np.asarray(image)
    images_arrays.append(image)


average_image = np.mean(images_arrays, axis = 0)

os.chdir(os.path.dirname(os.path.dirname(os.getcwd()))+"/Data_Analysis")

if gray_scaled:
    plt.imshow(average_image.astype('uint8'), cmap=plt.get_cmap('gray'))
    plt.savefig(f"Grayscale_{amount}.png")
else:
   plt.imshow(average_image.astype('uint8'))
   plt.savefig(f"Colours_{amount}.png")
    
plt.show()


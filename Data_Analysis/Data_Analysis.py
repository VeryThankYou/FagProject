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
logupvotes = np.log(upvotes+1)

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

data_description_log = {}
data_description_log["Min"] = np.min(logupvotes)
data_description_log["Max"] = np.max(logupvotes)
data_description_log["Mean"] = round(np.mean(logupvotes),1)
data_description_log["Std"] = round(np.std(logupvotes),1)
data_description_log["Q_1"] = np.quantile(logupvotes, q=0.25)
data_description_log["Median"] = np.median(logupvotes)
data_description_log["Q_3"] = np.quantile(logupvotes, q=0.75)

data_description = pd.DataFrame(data_description, index=[0])
print(data_description.style.to_latex(position = "H", position_float="centering", hrules = True))

data_description_log = pd.DataFrame(data_description_log, index=[0])
data_description_log = data_description_log.style.format(decimal='.', thousands=',', precision=2)
print(data_description_log.to_latex(position = "H", position_float="centering", hrules = True))


# Histograms

os.chdir(os.path.dirname(os.getcwd())+'/Data_Analysis')
plt.hist(upvotes, bins = 20) #Histogram of distribution of upvotes
plt.title("Upvotes")
plt.savefig("Histogram_upvotes.png")
plt.show()


print(len(logupvotes))
plt.hist(logupvotes)
plt.savefig("Histogram_log_upvotes.png")
plt.show()


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


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

os.chdir('/Volumes/Seagate Expansion Drive/Clara/DTU/Fagprojekt')


df = pd.read_csv("submissions.csv")
upvotes = df["Score"]

data_description = {}

data_description["Min"] = np.min(upvotes)
data_description["Max"] = np.max(upvotes)
data_description["Mean"] = round(np.mean(upvotes),1)
data_description["Std"] = round(np.std(upvotes),1)
data_description["Mode"] = np.bincount(upvotes).argmax()

print(data_description)

plt.hist(upvotes, bins = 20)
plt.title("Upvotes")
plt.show()

data_description = pd.DataFrame(data_description, index=[0])
print(data_description.style.to_latex(position = "H", position_float="centering", hrules = True))


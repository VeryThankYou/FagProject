import transformers as tf
import torch
import numpy as np
import scipy.stats as ss
import pandas as pd
from PIL import Image
import os

# Change directory to get the dataset
origDir = os.getcwd()
os.chdir(origDir+"/Data")
# Load dataset, add logupvotes
df = pd.read_csv("submissions.csv")
upvotes = df["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
df["Logscore"] = logupvotes
ids = df["ID"]

# Rename id-column to the names of the images based on directory
def rename_ids():
    newids = []
    for e in ids:
        newids.append(os.getcwd() + "/resized_images256/EarthPorn-" + e + ".jpg")
    return newids

names = rename_ids()
df["Filename"] = names

# Load image processor
processor = tf.AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# Load test-data
testsplit = int(df.shape[0] / 20)
dftest = df.iloc[:testsplit]
X = [processor(Image.open(e)) for e in dftest["Filename"]]
ytrain = df.iloc[testsplit:]["Logscore"].to_numpy()
y = dftest["Logscore"].to_numpy()

# Change directory back
os.chdir(origDir + "/ResNet")

# Find latest snapshot of the trained model
# If trained locally, the folder name needs to be "resultsCNN" instead of "resultsHPC"
lastCheckpoint = 0
for filename in os.listdir(os.getcwd() + "/resultsHPC"):
    f = os.path.join(os.getcwd() + "/resultsHPC", filename)
    num = int(f.split("-")[-1])
    if num > lastCheckpoint:
        lastCheckpoint = num

# Load model based on the snapshot
model = tf.ResNetForImageClassification.from_pretrained("./resultsHPC/checkpoint-"+str(lastCheckpoint), num_labels = 1, ignore_mismatched_sizes = True).to("cuda")

# Load inputs from test data
inputs = [torch.reshape(torch.as_tensor(pic["pixel_values"][0]).float(), (-1, 3, 224, 224)).cuda() for pic in X]

# Get the model's predictions
modelpred = [float(model(pic).logits) for pic in inputs]

# Get the baseline model's prediction (mean of training output)
BLpred = [np.mean(ytrain) for i in range(len(y))]

# Perform statistical test
zobs = [abs(modelpred[i] - y[i]) - abs(BLpred[i] - y[i]) for i in range(len(y))]
zmean = np.mean(zobs)
zvar = np.var(zobs)
zstd = np.std(zobs)
alpha = 0.01

p = 2 * ss.t.cdf(-abs(zmean), df = len(y) - 1, loc = 0, scale = zstd)
print(p)
print(zmean)
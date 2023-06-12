import transformers as tf
import torch
import numpy as np
import scipy.stats as ss
import pandas as pd
from PIL import Image
from torchvision import utils

df = pd.read_csv("submissions.csv")
upvotes = df["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
df["Logscore"] = logupvotes
ids = df["ID"]

def rename_ids():
    newids = []
    for e in ids:
        newids.append("resized_images/EarthPorn-" + e + ".png")
    return newids

names = rename_ids()
testsplit = int(df.shape[0] / 20)
df["Filename"] = names
dftest = df.iloc[:testsplit]

processor = tf.AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = tf.ResNetForImageClassification.from_pretrained("./resultsHPC/checkpoint-365100", num_labels = 1, ignore_mismatched_sizes = True).to("cuda")

X = [processor(Image.open(e)) for e in dftest["Filename"]]
ytrain = df.iloc[testsplit:]["Logscore"].to_numpy()
y = dftest["Logscore"].to_numpy()

inputs = [torch.reshape(torch.as_tensor(pic["pixel_values"][0]).float(), (-1, 3, 224, 224)).cuda() for pic in X]

modelpred = [float(model(pic).logits) for pic in inputs]

BLpred = [np.mean(ytrain) for i in range(len(y))]

zobs = [abs(modelpred[i] - y[i]) - abs(BLpred[i] - y[i]) for i in range(len(y))]
zmean = np.mean(zobs)
zvar = np.var(zobs)
zstd = np.std(zobs)
alpha = 0.01

p = 2 * ss.t.cdf(-abs(zmean), df = len(y) - 1, loc = 0, scale = zstd)
print(p)

print("hej")
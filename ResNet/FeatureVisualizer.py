import transformers as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
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
df20 = df.iloc[:20]

# Change directory back
os.chdir(origDir + "/ResNet")

# Load image processor
processor = tf.AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# Find latest snapshot of the trained model
# If trained locally, the folder name needs to be "resultsCNN" instead of "resultsHPC"
lastCheckpoint = 0
for filename in os.listdir(os.getcwd() + "/resultsHPC"):
    f = os.path.join(os.getcwd() + "/resultsHPC", filename)
    num = int(f.split("-")[-1])
    if num > lastCheckpoint:
        lastCheckpoint = num
model = tf.ResNetForImageClassification.from_pretrained("./resultsHPC/checkpoint-"+str(lastCheckpoint), num_labels = 1, ignore_mismatched_sizes = True).to("cuda")

# Load images with highest and lowest amount of upvotes
bestPic = df[df['Logscore']==df['Logscore'].max()]
worstPic = df[df['Logscore']==df['Logscore'].min()]

# Load the image to perform feature extraction on as an input tensor
# Change bestPic to worstPic to change the image
dummyX = torch.reshape(torch.as_tensor(processor(Image.open(bestPic["Filename"][bestPic.ID.index[0]]))["pixel_values"][0]).float(), (-1, 3, 224, 224)).cuda()

# Show the processed image
plt.imshow(torch.Tensor.cpu(dummyX[0].permute(1, 2, 0)))
plt.show()
plt.clf()

# Print the model so the layers to extract features from can be chosen
print(model)

# Choose a couple of layers
# They is converted to CPU-tensors instead of CUDA-tensors, as CUDA-tensors can't be visualized
firstLayerOutput = model.resnet.embedder(dummyX)
filters = torch.Tensor.cpu(firstLayerOutput)
secondLayerOutput = model.resnet.encoder.stages[0](firstLayerOutput)
filters2 = torch.Tensor.cpu(secondLayerOutput)
thirdLayerOutput = model.resnet.encoder.stages[1](secondLayerOutput)
filters3 = torch.Tensor.cpu(thirdLayerOutput)
fourthLayerOutput = model.resnet.encoder.stages[2](thirdLayerOutput)
filters4 = torch.Tensor.cpu(fourthLayerOutput)
firstPointFifthLayerOutput = model.resnet.encoder.stages[0].layers[0](firstLayerOutput)
filters1_5 = torch.Tensor.cpu(firstPointFifthLayerOutput)

print("Shapes of filters: " + str(filters.shape) + ", " + str(filters2.shape) + ", " + str(filters3.shape) + ", " + str(filters4.shape) + ", " + str(filters1_5.shape))

# Set directory
os.chdir(os.path.dirname(os.getcwd())+'/ResNet')

# Show and save specific feature channels
plt.imshow(filters[0,  25, :, :].detach().numpy(), cmap='gray')
plt.savefig("ExtractedFeatures/BestPicEmbLayFilter26.png")
plt.show()
plt.clf()

plt.imshow(filters[0,  50, :, :].detach().numpy(), cmap='gray')
plt.savefig("ExtractedFeatures/BestPicEmbLayFilter51.png")
plt.show()
plt.clf()

plt.imshow(filters[0,  63, :, :].detach().numpy(), cmap='gray')
plt.savefig("ExtractedFeatures/BestPicEmbLayFilter64.png")
plt.show()
plt.clf()


# From here on grids of feature channels are shown and saved depending on which lines are outcommented.
square = 4

"""
for screen in range(4):
    ix = 1
    
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(filters[0,  screen * 16 + ix-1, :, :].detach().numpy(), cmap='gray')
            ix += 1
            # show the figure
    plt.suptitle("Embedding layer features, page " + str(screen + 1) + "/4")
    plt.savefig("ExtractedFeatures/BestPicEmbLay" + str(screen + 1) + "_4.png")
    #plt.show()
    plt.clf()
"""

"""
for screen in range(16):
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(filters2[0,  screen * 16 + ix-1, :, :].detach().numpy(), cmap='gray')
            ix += 1
            # show the figure
    plt.suptitle("First ResNet stage features, page " + str(screen + 1) + "/16")
    plt.savefig("ExtractedFeatures/BestPicEncLay1_" + str(screen + 1) + "_16.png")
    #plt.show()
    plt.clf()
"""

for screen in range(16):
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(filters1_5[0,  screen * 16 + ix-1, :, :].detach().numpy(), cmap='gray')
            ix += 1
            # show the figure
    plt.suptitle("First ResNet block features, page " + str(screen + 1) + "/16")
    #plt.savefig("ExtractedFeatures/WorstPicEncBlock1_" + str(screen + 1) + "_16.png")
    #plt.show()
    plt.clf()

for screen in range(32):
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(filters3[0,  screen * 16 + ix-1, :, :].detach().numpy(), cmap='gray')
            ix += 1
            # show the figure
    plt.suptitle("Second ResNet stage features, page " + str(screen + 1) + "/32")
    #plt.show()
    plt.clf()

for screen in range(64):
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(filters4[0,  screen * 16 + ix-1, :, :].detach().numpy(), cmap='gray')
            ix += 1
            # show the figure
    plt.suptitle("Second ResNet stage features, page " + str(screen + 1) + "/64")
    #plt.show()
    plt.clf()

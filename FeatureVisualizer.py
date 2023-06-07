from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import transformers as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers.utils.fx import symbolic_trace
import pandas as pd
from PIL import Image
from torchvision import utils

df = pd.read_csv("submissions.csv")
upvotes = df["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
df["Logscore"] = logupvotes
ids = df["ID"]

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))   

def rename_ids():
    newids = []
    for e in ids:
        newids.append("resized_images/EarthPorn-" + e + ".png")
    return newids

names = rename_ids()
df["Filename"] = names
df20 = df.iloc[:20]

processor = tf.AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = tf.ResNetForImageClassification.from_pretrained("./results/checkpoint-39600", num_labels = 1, ignore_mismatched_sizes = True).to("cuda")

dummyX = torch.reshape(torch.as_tensor(processor(Image.open(df20["Filename"][0]))["pixel_values"][0]).float(), (-1, 3, 224, 224)).cuda()
plt.imshow(torch.Tensor.cpu(dummyX[0].permute(1, 2, 0)))
plt.show()
plt.clf()

print(model)
firstLayerOutput = model.resnet.embedder(dummyX)
filters = torch.Tensor.cpu(firstLayerOutput)
secondLayerOutput = model.resnet.encoder.stages[0](firstLayerOutput)
filters2 = torch.Tensor.cpu(secondLayerOutput)

square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(filters[0,  ix-1, :, :].detach().numpy(), cmap='gray')
        ix += 1
        # show the figure
plt.show()
plt.clf()

plt.imshow(filters2[0,  0, :, :].detach().numpy(), cmap='gray')
plt.show()
# Code for saving model as torch model
#dummyX = torch.reshape(torch.as_tensor(processor(Image.open(df20["Filename"][0]))["pixel_values"][0]).float(), (-1, 3, 224, 224)).cuda()
#y = df20["Logscore"].to_numpy()
#model.eval()
#traced_model = torch.jit.trace(model, dummyX)
#torch.jit.save(traced_model, "traced_CNN.pt")



feature_extractor = create_feature_extractor(model, return_nodes=['.resnet.embedder'])

#nodes, _ = get_graph_node_names(model)

print(model)

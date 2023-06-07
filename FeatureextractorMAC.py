import numpy as np
import pandas as pd
import PIL
import transformers as tf
import torch
from PIL import Image
import requests
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import utils, transforms

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))   

class CustomImageDataset(Dataset):
    def __init__(self, Xs, ys):
        self.img_labels = ys
        self.inputs = Xs


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = {"pixel_values": torch.as_tensor(self.inputs[idx]["pixel_values"][0]).float()}
        image["labels"] = torch.as_tensor(self.img_labels[idx]).float()
        return image

processor = tf.AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = tf.ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels = 1, ignore_mismatched_sizes = True)

model.eval()
image = Image.open('./resized_images/EarthPorn-1a0gxo.png').convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = preprocess(image)
input_batch = input_image.unsqueeze(0)

with torch.no_grad():
    intermediate_layer_outputs = model.conv1(input_batch)
    for layer in model.layer1:
        intermediate_layer_outputs = layer(intermediate_layer_outputs)
    for layer in model.layer2:
        intermediate_layer_outputs = layer(intermediate_layer_outputs)
    for layer in model.layer3:
        intermediate_layer_outputs = layer(intermediate_layer_outputs)
    for layer in model.layer4:
        intermediate_layer_outputs = layer(intermediate_layer_outputs)

desired_layer_output = intermediate_layer_outputs

feature_vector = desired_layer_output.view(desired_layer_output.size(0), -1)
import torch
from torch import nn


class ConvBlock(nn.Module):
   """
   Applies `num_layers` 3x3 convolutions each followed by ReLU then downsamples
   via 2x2 max pool.
   """

   def __init__(self, num_layers, in_channels, out_channels):
       super().__init__()
       self.convs = nn.ModuleList(
           [nn.Sequential(
               nn.Conv2d(in_channels if i==0 else out_channels, out_channels, 3, padding=1),
               nn.ReLU()
            )
            for i in range(num_layers)]
       )
       self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
      
   def forward(self, x):
       for conv in self.convs:
           x = conv(x)
       x = self.downsample(x)
       return x
      

class CNN(nn.Module):
   """
   Applies several ConvBlocks each doubling the number of channels, and
   halving the feature map size, before taking a global average and classifying.
   """

   def __init__(self, in_channels, num_blocks, num_classes):
       super().__init__()
       first_channels = 64
       self.blocks = nn.ModuleList(
           [ConvBlock(
               2 if i==0 else 3,
               in_channels=(in_channels if i == 0 else first_channels*(2**(i-1))),
               out_channels=first_channels*(2**i))
            for i in range(num_blocks)]
       )
       self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
       self.cls = nn.Linear(first_channels*(2**(num_blocks-1)), num_classes)

   def forward(self, x):
       for block in self.blocks:
           x = block(x)
       x = self.global_pool(x)
       x = x.flatten(1)
       x = self.cls(x)
       return x


model = CNN(3, 4, 10)
out = model(torch.zeros(1, 3, 32, 32))  # This will be the final logits over classes

from torchvision.models.feature_extraction import create_feature_extractor
# Confused about the node specification here?
# We are allowed to provide truncated node names, and `create_feature_extractor`
# will choose the last node with that prefix.
feature_extractor = create_feature_extractor(
	model, return_nodes=['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3'])
# `out` will be a dict of Tensors, each representing a feature map
out = feature_extractor(torch.zeros(1, 3, 32, 32))
print(out)
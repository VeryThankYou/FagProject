# load vgg model
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

import pandas as pd
import numpy as np


from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

# load the model
model = ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
# summarize the model
model = Model(inputs=model.inputs, outputs=model.layers[2].output)
model.summary()

img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = pyplot.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        ix += 1
pyplot.show()

#############################################
# Data
data = pd.read_csv("submissions.csv")
logupvotes = np.log(data["Score"].to_numpy()+1)
data["Logscore"] = logupvotes

def rename_ids(ids):
    newids = []
    for e in ids:
        name = "resized_images/EarthPorn-" + e + ".png"
        newids.append(name)
    return newids
result = rename_ids(data['ID'])
data["Filename"] = result

np_pictures = np.zeros((len(result),600,600,3))

for i, file in enumerate(data["Filename"]):
    image = load_img(file)
    # convert image to numpy array
    np_pictures[i] = np.asarray(image)   
    print(i)
datagen = ImageDataGenerator(
    rescale = 1./255)



df1000 = data.iloc[:1000]
X = datagen.fit(df1000["Pictures"])
y = df1000["Logscore"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)



# Get ResNet-50 Model
def getResNet50Model(lastFourTrainable=False):
  resnet_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=True)

  # Make all layers non-trainable
  for layer in resnet_model.layers[:]:
      layer.trainable = False

  # Add fully connected layer which have 1024 neuron to ResNet-50 model
  output = resnet_model.get_layer('avg_pool').output
  output = Flatten(name='new_flatten')(output)
  output = Dense(units=1024, activation='relu', name='new_fc')(output)
  output = Dense(units=10, activation='softmax')(output)
  resnet_model = Model(resnet_model.input, output)

  # Make last 4 layers trainable if lastFourTrainable == True
  if lastFourTrainable == True:
    resnet_model.get_layer('conv5_block3_2_bn').trainable = True
    resnet_model.get_layer('conv5_block3_3_conv').trainable = True
    resnet_model.get_layer('conv5_block3_3_bn').trainable = True
    resnet_model.get_layer('new_fc').trainable = True

  # Compile ResNet-50 model
  resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  resnet_model.summary()
  
  return resnet_model
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import imageio

from PIL import Image

import os
os.chdir("/Volumes/Seagate Expansion Drive/Clara/DTU/Fagprojekt")
submissions = pd.read_csv('submissions.csv')

import subprocess
import os

# RUN ONLY THE FIRST TIME: Clone Real-ESRGAN and enter the Real-ESRGAN
"""command = 'git clone https://github.com/xinntao/Real-ESRGAN.git'
subprocess.run(command, shell=True)"""
os.chdir("./Real-ESRGAN")

# RUN ONLT THE FIRST TIME: Set up environment
"""
command = 'pip install basicsr'
subprocess.run(command, shell=True)
command = 'pip install facexlib'
subprocess.run(command, shell=True)
command = 'pip install gfpgan'
subprocess.run(command, shell=True)
command = 'pip install -r requirements.txt'
subprocess.run(command, shell=True)
command = 'python setup.py develop'
subprocess.run(command, shell=True)
"""

import os
import shutil

upload_folder = 'inputs'
result_folder = 'results'

command = 'python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --outscale 3.5 --face_enhance -o results'
subprocess.run(command, shell=True)

# utils for visualization
import cv2
import matplotlib.pyplot as plt
def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1) 
  plt.title('Input image', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('Real-ESRGAN output', fontsize=16)
  
  ax2.axis('off')
  ax1.imshow(img1)
  ax2.imshow(img2)
def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

# display each image in the upload folder
import os
import glob

input_list = sorted(glob.glob(os.path.join(upload_folder, '*')))
output_list = sorted(glob.glob(os.path.join(result_folder, '*')))
for input_path, output_path in zip(input_list, output_list):
  img_input = imread(input_path)
  img_output = imread(output_path)
  display(img_input, img_output)



import torch
import torchvision
import requests
import tqdm
import pyspng
import ninja 
import imageio
import subprocess
import os

print(os.environ.get('CUDA_PATH'))
os.chdir("./stylegan3-main")
command = 'python train.py --outdir=./StyleGAN_results --cfg=stylegan3-r --data=./datasets/resized_images512.zip --gpus=1 --batch=16 --gamma=6.6 --mirror=1 --kimg=5 --snap=1 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl'

subprocess.run(command, shell=True)

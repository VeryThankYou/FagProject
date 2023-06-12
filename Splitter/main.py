from split_image import split_image
import sys
from PIL import Image
import os
split_image("all.png",16,30,False,False,output_dir="./images")
split_image("best.png",16,30,False,False,output_dir="./images")
split_image("start.png",16,30,False,False,output_dir="./images")
split_image("worst.png",16,30,False,False,output_dir="./images")

for i in range(30*16):
    images = [Image.open(x) for x in ['./images/start_'+str(i)+'.png','./images/worst_'+str(i)+'.png','./images/all_'+str(i)+'.png','./images/best_'+str(i)+'.png' ]]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('./results/combined'+str(i)+'.png')
from split_image import split_image
import sys
from PIL import Image
import random
for i in range(31):
    name="./data/combined"+str(i)+".png"
    split_image(name,1,4,False,False,output_dir="./images")
#Order is 0=pretrained, 1=lower upvotes, 2=all data, 3=higher upvotes
file=open("order.txt","a")

for i in range(31):
    temp=[]
    for j in range(4):
        temp.append("./images/combined"+str(i)+"_"+str((j))+".png")
    random.shuffle(temp)
    order=""
    for j in range(4):
        order+=str(temp[j])+":"
    file.write(order)
    file.write("\n")
    images = [Image.open(x) for x in temp]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('./results/combined'+str(i)+'.png')    


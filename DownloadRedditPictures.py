import praw
import requests
import cv2
import numpy as np
import os
import pickle
import time
import multiprocessing
from numba import jit, cuda
from datetime import timedelta
from utils.create_token import create_token
class DownloadRedditPictures:
    def __init__(self,search,resize):
        self.search=search
        self.resize=resize
    def create_folder(self,image_path):
        CHECK_FOLDER = os.path.isdir(image_path)
        if not CHECK_FOLDER:
            os.makedirs(image_path)
    def savePictures(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        image_path = os.path.join(dir_path, "images/")
        ignore_path = os.path.join(dir_path, "ignore_images/")
        self.create_folder(image_path)
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        else:
            creds = create_token()
            pickle_out = open("token.pickle","wb")
            pickle.dump(creds, pickle_out)

        reddit = praw.Reddit(client_id=creds['client_id'],
                            client_secret=creds['client_secret'],
                            user_agent=creds['user_agent'],
                            username=creds['username'],
                            password=creds['password'])


        f_final = open("sub_list.csv", "r")
        img_notfound = cv2.imread('imageNF.png')
        startTime=time.time()
        for line in f_final:
            sub = line.strip()
            subreddit = reddit.subreddit(sub)

            print(f"Starting {sub}!")
            totalcount=0
            count = 0
            failed=0
            for submission in subreddit.new(limit=self.search):
                if "jpg" in submission.url.lower() or "png" in submission.url.lower():
                    try:
                        if((int(submission.preview["images"][0]["resolutions"][5]["width"])<self.resize) or int(submission.preview["images"][0]["resolutions"][5]["height"])<self.resize):
                            resp= requests.get(submission.url.lower(), stream=True).raw
                            count+=1
                        else:
                            resp = requests.get(submission.preview["images"][0]["resolutions"][5]["url"], stream=True).raw
                        image = np.asarray(bytearray(resp.read()), dtype="uint8")
                        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                        compare_image=cv2.resize(image,(224,224))
                        # Get all images to ignore
                        for (dirpath, dirnames, filenames) in os.walk(ignore_path):
                            ignore_paths = [os.path.join(dirpath, file) for file in filenames]
                        ignore_flag = False

                        for ignore in ignore_paths:
                            ignore = cv2.imread(ignore)
                            difference = cv2.subtract(ignore, compare_image)
                            b, g, r = cv2.split(difference)
                            total_difference = cv2.countNonZero(b) + cv2.countNonZero(g) + cv2.countNonZero(r)
                            if total_difference == 0:
                                totalcount+=1
                                failed+=1
                                print("Status: " +str(totalcount)+"/"+str(self.search))   
                                ignore_flag = True



                        if not ignore_flag:
                            cv2.imwrite(f"{image_path}{sub}-{submission.id}.png", image)
                            totalcount+=1
                            print("Status: " +str(totalcount)+"/"+str(self.search))   
                            
                    except Exception as e:
                        totalcount+=1
                        failed+=1
        endTime=int(time.time()-startTime)
        td=timedelta(seconds=endTime)
        print("Time elapsed in hh:mm:ss | "+str(td))
        print("Images raw: "+str(count))
        print("Images failed to save: "+str(failed))    

# Create directory if it doesn't exist to save images


# Path to save images


# Get token file to log into reddit.
# You must enter your....
# client_id - client secret - user_agent - username password


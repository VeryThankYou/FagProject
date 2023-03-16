import praw
import requests
import cv2
import numpy as np
import os
import pickle
import time

from numba import jit, cuda
from datetime import timedelta
from utils.create_token import create_token
from postDownloader import download_from_url
from datetime import datetime
class DownloadRedditPictures:

    def __init__(self,resize):
        self.resize=resize
        start_time = datetime.utcnow()  #datetime.strptime("10/05/2021", "%m/%d/%Y")
        end_time = datetime.strptime("12/31/2011", "%m/%d/%Y")  #datetime.strptime("09/25/2021", "%m/%d/%Y")
        self.ids=download_from_url(start_time,end_time)
        self.ids = [i if i.startswith('t3_') else f't3_{i}' for i in self.ids]
        self.search=int(len(self.ids))
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

        self.reddit = praw.Reddit(client_id=creds['client_id'],
                            client_secret=creds['client_secret'],
                            user_agent=creds['user_agent'],
                            username=creds['username'],
                            password=creds['password'])
        sub="EarthPorn"
        self.savePictures(sub,image_path,ignore_path)

    def create_folder(self,image_path):
        CHECK_FOLDER = os.path.isdir(image_path)
        if not CHECK_FOLDER:
            os.makedirs(image_path)
    
    def savePictures(self,sub,image_path,ignore_path):
        startTime=time.time()
        count=0
        failedCount=0
        for submission in self.reddit.info(self.ids):
            if "jpg" in submission.url.lower() or "png" in submission.url.lower():
                try:
                    if((int(submission.preview["images"][0]["resolutions"][5]["width"])<self.resize) or int(submission.preview["images"][0]["resolutions"][5]["height"])<self.resize):
                        resp= requests.get(submission.url.lower(), stream=True).raw
                    else:
                        resp = requests.get(submission.preview["images"][0]["resolutions"][5]["url"], stream=True).raw
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    compare_image=cv2.resize(image,(224,224))

                    for (dirpath, dirnames, filenames) in os.walk(ignore_path):
                        ignore_paths = [os.path.join(dirpath, file) for file in filenames]
                    ignore_flag = False

                    for ignore in ignore_paths:
                        ignore = cv2.imread(ignore)
                        difference = cv2.subtract(ignore, compare_image)
                        b, g, r = cv2.split(difference)
                        total_difference = cv2.countNonZero(b) + cv2.countNonZero(g) + cv2.countNonZero(r)
                        if total_difference == 0:  
                            ignore_flag = True



                    if not ignore_flag:
                        cv2.imwrite(f"{image_path}{sub}-{submission.id}.png", image)
                        count+=1
                        print(count+"/" +len(self.ids))
                        endTime=len(time.time()-startTime)
                        td = timedelta(seconds=endTime)
                        print('Time elapsed in hh:mm:ss:', td)
                        
                except Exception as e:
                    count+=1
                    failedCount+=1
                    print(count+"/" +len(self.ids))
                    endTime=len(time.time()-startTime)
                    td = timedelta(seconds=endTime)
                    print('Time elapsed in hh:mm:ss:', td)
        print("In total: "+failedCount +"pictures failed to save!")
        endTime=len(time.time()-startTime)
        td = timedelta(seconds=endTime)
        print('Time taken to finish in hh:mm:ss:', td)
from DownloadRedditPictures import DownloadRedditPictures
from Image_Resizer import resize_all

size=600
limit=100

redditPictures=DownloadRedditPictures(limit,size)
redditPictures.savePictures()
resize_all(size)
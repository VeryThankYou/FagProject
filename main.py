from DownloadRedditPictures import DownloadRedditPictures
from Image_Resizer import resize_all

size=600
limit=1000
redditPictures=DownloadRedditPictures(limit,size)
redditPictures.savePictures()
resize_all(size)
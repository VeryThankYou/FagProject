from DownloadRedditPictures import DownloadRedditPictures
from Image_Resizer import resize_all

size=600
redditPictures=DownloadRedditPictures(size)
resize_all(size)
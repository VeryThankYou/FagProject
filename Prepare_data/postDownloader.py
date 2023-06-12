import requests
from datetime import datetime
import traceback
import time
import json
import sys
import csv
import json


# default start time is the current time and default end time is all history
# you can change out the below lines to set a custom start and end date. The script works backwards, so the end date has to be before the start date
start_time = datetime.utcnow()  #datetime.strptime("10/05/2021", "%m/%d/%Y")
end_time = datetime.strptime("12/31/2022", "%m/%d/%Y")  #datetime.strptime("09/25/2021", "%m/%d/%Y")


def download_from_url(start_datetime, end_datetime):
	print(f"Saving:")

	ids = []
	url_template = "https://api.pushshift.io/reddit/{}/search?limit=1000&order=desc&{}&before="
	url_base = url_template.format("submission", "subreddit=EarthPorn")

	previous_epoch = int(start_datetime.timestamp())
	break_out = False
	count = 0
	while True:
		new_url = url_base+str(previous_epoch)
		json_text = requests.get(new_url, headers={'User-Agent': "Post downloader by /u/Watchful1"})
		time.sleep(1)  # pushshift has a rate limit, if we send requests too fast it will start returning error messages
		try:
			json_data = json_text.json()
		except json.decoder.JSONDecodeError:
			time.sleep(1)
			continue

		if 'data' not in json_data:
			break
		objects = json_data['data']
		if len(objects) == 0:
			break

		for obj in objects:
			previous_epoch = obj['created_utc'] - 1
			if end_datetime is not None and datetime.utcfromtimestamp(previous_epoch) < end_datetime:
				break_out = True
				break
			ids.append(str(obj['id']))
			count += 1
			
		if break_out:
			break

		print(f"Saved {count} through {datetime.fromtimestamp(previous_epoch).strftime('%Y-%m-%d')}")

	print(f"Saved {count}")
	return ids


if __name__ == "__main__":
	ids = download_from_url(start_time, end_time)
	print(len(ids))

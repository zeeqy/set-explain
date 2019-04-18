import json, sys, os, time
import argparse
import threading
import requests
import urllib

"""

Extract wiki document title from wiki dump json

"""
def monthly_view(data):
	if "items" not in data.keys():
		return -1
	else:
		views = 0
		for res in data['items']:
			if 'views' in res.keys():
				views += res['views']
		return views // 12

def merge_task(task_list, args):
	for folder in task_list:
		view_freq = {}
		outputname = 'FREQ_{}'.format(folder) 
		working_dir = '{}/{}'.format(args.input_dir,folder)
		for fname in os.listdir(working_dir):
			with open('{}/{}'.format(working_dir,fname), 'r', errors='ignore') as f:
				raw = f.readlines()
			f.close()
			for item in raw:
				item_dict = json.loads(item)
				title = item_dict['title']
				enc_title = urllib.parse.quote(title)
				url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{}/monthly/20180101/20190101".format(enc_title)
				r = requests.get(url)
				avg_view = monthly_view(r.json())
				view_freq.update({title: avg_view})
			print("finish requesting {}/{}".format(folder, fname))
		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write(json.dumps(view_freq))
		f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
	parser = argparse.ArgumentParser(description="Extract title from wiki json")
	parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
	
	args = parser.parse_args()

	dump_dir = os.listdir(args.input_dir)
	tasks = list(split(dump_dir, args.num_process))

	threads = []
	for i in range(args.num_process):
		t = threading.Thread(target=merge_task, args=(tasks[i], args,))
		threads.append(t)
		t.start()

if __name__ == '__main__':
	main()
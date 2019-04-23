from aiohttp import ClientSession, TCPConnector
import asyncio
import sys, os, time
from pypeln import asyncio_task as aio
import urllib
import argparse
import json

async def fetch(url, session):
	async with session.get(url) as response:
		res = await response.read()
		jres = json.loads(res)
		views = 0
		if 'items' in jres.keys():
			for v in jres['items']:
				views += v['views']
			print(views//12)


def main():

	parser = argparse.ArgumentParser(description="Page views collect")
	parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	args = parser.parse_args()

	dump_dir = os.listdir(args.input_dir)

	limit = 80

	for folder in dump_dir:
		titles = []
		working_dir = '{}/{}'.format(args.input_dir,folder)
		for fname in os.listdir(working_dir):
			with open('{}/{}'.format(working_dir,fname), 'r', errors='ignore') as f:
				raw = f.readlines()
			f.close()
			
			for item in raw:
				item_dict = json.loads(item)
				titles.append(item_dict['title'])
	print(len(titles))
	chunks = [titles[i:i + 80] for i in range(0, len(titles), 80)]
	for c in chunks:
		urls = ["https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{}/monthly/20180101/20190101".format(urllib.parse.quote(t)) for t in c]
		aio.each(
			fetch, 
			urls,
			workers = limit,
			on_start = lambda: ClientSession(connector=TCPConnector(limit=None)),
			on_done = lambda _status, session: session.close(),
			run = True,
		)
		time.sleep(0.2)

if __name__ == '__main__':
	main()
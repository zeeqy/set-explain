from aiohttp import ClientSession, TCPConnector
import asyncio
import sys, os, time
from pypeln import asyncio_task as aio
import urllib
import argparse
import json

freq_collect = {}

async def fetch(t, session):
	url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{}/monthly/20180101/20190101".format(urllib.parse.quote(t))
	async with session.get(url) as response:
		res = await response.read()
		jres = json.loads(res)
		views = 0
		if 'items' in jres.keys():
			for v in jres['items']:
				views += v['views']
			freq_collect.update({t:views // 12})
		else:
			freq_collect.update({t:-1})


def main():

	parser = argparse.ArgumentParser(description="Page views collect")
	parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	args = parser.parse_args()

	limit = 80

	fname = args.input_dir.split('/')[-1]
	
	with open(args.input_dir, 'r') as f:
		raw = f.read()
	f.close()

	titles = raw.split('\n')

	chunks = [titles[i:i + 80] for i in range(0, len(titles), 80)]
	for c in chunks:
		aio.each(
			fetch, 
			c,
			workers = limit,
			on_start = lambda: ClientSession(connector=TCPConnector(limit=None)),
			on_done = lambda _status, session: session.close(),
			run = True,
		)
		time.sleep(0.2)

	with open('{}/FREQ_{}'.format(args.output_dir,fname), 'w+') as f:
		f.write(json.dumps(freq_collect))
	f.close()

	return 0

if __name__ == '__main__':
	main()
	
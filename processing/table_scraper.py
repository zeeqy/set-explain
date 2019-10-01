from aiohttp import ClientSession, TCPConnector
import asyncio
import sys, os, time
from pypeln import asyncio_task as aio
import urllib
import argparse
import json
import re
import json
import time
from bs4 import BeautifulSoup

freq_collect = []
entityset = set()

async def fetch(t, session):
	url = 'https://en.wikipedia.org/wiki/{}'.format(t.replace(' ', '_'))
	title_text = t[8:]
	async with session.get(url) as response:
		res = await response.read()
		soup = BeautifulSoup(res,'lxml')
		_table = soup.find('table',{'class':'wikitable'})
		if _table is None or _table.tbody is None:
			return
		ent = []
		for row in _table.tbody.findAll('tr')[1:]:
			if len(row.findAll('td')) == 0:
				continue
			first_column = row.findAll('td')[0]
			try:
				string = first_column.find('a').text
			except:
				continue
			if string is None:
				continue
			if '(page does not exist)' in string:
				continue
			else:
				string = re.sub(r'\([^)]*\)', '', string)
				try: 
					string = string.encode('ascii').decode()
				except:
					continue
				string = string.lower()
				if string.split(',')[0] in entityset:
					ent.append(string.split(',')[0])
		if len(ent) >= 10:
			tid = np.random.randint(10000, high=99999, size=None, dtype='int64')
			freq_collect.append({'id':'WL{}'.format(tid), 'title':title_text, "entities":ent, 'url':url})

def main():

	parser = argparse.ArgumentParser(description="Page table scraper")
	parser.add_argument('--entity_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--title_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	args = parser.parse_args()

	with open('{}/wiki_quality.txt'.format(args.entity_dir), 'r') as f:
		raw_list = f.read()
	f.close()
	entityset = set(raw_list.split('\n'))

	with open('{}/list_pages.txt'.format(args.title_dir), 'r') as f:
		raw_list = f.read()
	f.close()
	wikititle = raw_list.split('\n')

	limit = 80

	chunks = [wikititle[i:i + 80] for i in range(0, len(wikititle), 80)]
	i = 1
	for c in chunks:
		aio.each(
			fetch, 
			c,
			workers = limit,
			on_start = lambda: ClientSession(connector=TCPConnector(limit=None)),
			on_done = lambda _status, session: session.close(),
			run = True,
		)
		time.sleep(1)

		with open('{}/SCRAPER_{}'.format(args.output_dir,i), 'w+') as f:
			f.write(json.dumps(freq_collect))
		f.close()

		freq_collect = []
		i += 1

	return 0

if __name__ == '__main__':
	main()
	
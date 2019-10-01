import multiprocessing as mp
from multiprocessing import Pool
import sys, os, time
import requests
import argparse
from tqdm import tqdm
import json
import re
import json
import time
from bs4 import BeautifulSoup

def scraper(params):
    wikititle, entityset, pid = params

    tables = []
    tid = 1
    for title in tqdm(wikititle, desc='processing-{}'.format(pid), mininterval=10):
        url = 'https://en.wikipedia.org/wiki/{}'.format(title.replace(' ', '_'))
        website_url = requests.get(url).text
        title_text = title[8:]
        soup = BeautifulSoup(website_url,'lxml')
        My_table = soup.find('table',{'class':'wikitable'})
        if My_table is None or My_table.tbody is None:
            continue
        ent = []
        for row in My_table.tbody.findAll('tr')[1:]:
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
            tables.append({'id':'SCR{}{}'.format(pid,tid), 'title':title_text, "entities":ent, 'url':url})
            tid +=1
        
    if len(tables) != 0:
        with open('/mnt/nfs/work1/allan/zhiqihuang/set-explain/data/SCRAPER_{}'.format(pid+1), 'w+') as f:
            for s in tables:
                f.write(json.dumps(s)+'\n')
        f.close()


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

with open('/mnt/nfs/work1/allan/zhiqihuang/HiExpan/src/tools/AutoPhrase/data/EN/wiki_quality.txt', 'r') as f:
    raw_list = f.read()
f.close()
entityset = set(raw_list.split('\n'))

with open('/mnt/nfs/work1/allan/zhiqihuang/set-explain/data/list_pages.txt', 'r') as f:
    raw_list = f.read()
f.close()

wikititle = raw_list.split('\n')

tasks = list(split(wikititle, 10))

inputs = [(tasks[i], entityset, i) for i in range(num_process)]

with Pool(10) as p:
    p.map(scraper, inputs)
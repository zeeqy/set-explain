import json, sys, os, re, time
import argparse
import bisect
import multiprocessing as mp
from tqdm import tqdm
from bs4 import BeautifulSoup
from wikitables import import_tables

"""
keep json format and find list pages

"""

def merge_task(task_list, args, pid):
	context = []
	outputname = 'LIST-{}'.format(pid)
	for wt in tqdm(task_list, desc='pages-{}'.format(pid), mininterval=30):
		tables = import_tables(wt)
		if len(tables) > 1:
			title = parse(wt[8:].lower())
			if title != '':
				ent_list = []
				for row in tables[0].rows:
					key = list(row.keys())[0]
					ent = parse(row[key].lower())
					if ent != '':
						ent_list.append(ent)
				if len(ent_list) >= 5:
					context.append(context.append({'title':title, 'ents':ent_list}))
		time.sleep(0.1)
	
	if context != []:
		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join(context))
		f.close()

	# for folder in task_list:
	# 	outputname = 'XML_{}'.format(folder)
	# 	working_dir = '{}/{}'.format(args.input_dir,folder)
	# 	context = []
	# 	for fname in os.listdir(working_dir):
	# 		with open('{}/{}'.format(working_dir,fname), 'r') as f:
	# 			raw = f.read()
	# 		f.close()
	# 		soup = BeautifulSoup(raw,'html')
	# 		docs = soup.find_all('doc')
	# 		for doc in tqdm(docs, desc='{}'.format(fname), mininterval=30):
	# 			if doc['title'][:7] != "List of":
	# 				continue # filter lists
	# 			else:
	# 				context.append(doc)

	# 	if context != []:
	# 		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
	# 			for doc in context:
	# 				f.write(str(doc))
	# 		f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def parse(text):
	new_text = text.replace('<br>','\n')
	new_text = re.sub(r'\([^)]*\)', '', new_text)
	new_text = new_text.replace('\n\n','\n')
	new_text = new_text.replace('<nowiki>','')
	new_text = new_text.replace('</nowiki>','')
	new_text = new_text.replace('<onlyinclude>','')
	new_text = new_text.replace('</onlyinclude>','')
	new_text = new_text.replace('()','')
	new_text = new_text.replace('  ',' ')
	new_text = new_text.replace('  ',' ')
	new_text = new_text.replace('\n\n','\n')
	new_text = re.sub(r'\[\[\bCategory:\b.*?\]\]', '', new_text)
	new_text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'\1', new_text)
	new_text = re.sub(r'\[\[\bFile:\b.*?\|\bthumb\b\|.*?\]\]\ ', '', new_text)
	new_text = new_text.replace('  ',' ')
	new_text = new_text.replace('  ',' ')
	new_text = new_text.replace('[...]','')
	new_text = new_text.replace('[]','')
	new_text = new_text.replace('[ ]','')
	new_text = new_text.replace('()','')
	new_text = new_text.replace('( )','')
	new_text = new_text.replace('  ',' ')
	new_text = new_text.replace('  ',' ')
	new_text = new_text.replace('\n\n','\n')
	new_text = re.sub(r'\([^)]*\)', '', new_text)
	new_text = re.sub(r'\s+([?.!,:; @+\-=<>{}#%^*()]|(\'s))', r'\1', new_text)
	new_text = re.sub(r'<.*?>', '', new_text)
	new_text.replace('(', '').replace(')','').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
	new_text = new_text.encode("ascii", errors="ignore").decode()
	return new_text.strip()

def main():
	parser = argparse.ArgumentParser(description="keep json format and find list pages")
	parser.add_argument('--list_pages', type=str, default='', help='list pages')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
	
	args = parser.parse_args()

	with open(args.list_pages, 'r') as f:
		list_pages = f.read()
	f.close()

	list_pages = list(set(list_pages.split('\n')))[:100]
	tasks = list(split(list_pages, args.num_process))

	processes = [mp.Process(target=merge_task, args=(tasks[i], args, i)) for i in range(args.num_process)]

	for p in processes:
		p.start()

	for p in processes:
		p.join()

if __name__ == '__main__':
	main()

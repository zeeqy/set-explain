import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from tqdm import tqdm
from bs4 import BeautifulSoup

"""
keep json format and find list pages

"""

def merge_task(task_list, args):
	for folder in task_list:
		outputname = 'XML_{}'.format(folder)
		working_dir = '{}/{}'.format(args.input_dir,folder)
		context = []
		for fname in os.listdir(working_dir):
			with open('{}/{}'.format(working_dir,fname), 'r') as f:
				raw = f.read()
			f.close()
			soup = BeautifulSoup(raw,'xml')
			docs = soup.find_all('doc')
			for doc in tqdm(docs, desc='{}'.format(fname), mininterval=30):
				if doc['title'][:7] != "List of":
					continue # filter lists
				else:
					title = doc['title'][7:].lower()
					ent_list = [item[12:].lower() for item in doc.text.split('\n') if len(item) > 12 and item[:12] == "BULLET::::- " and parse(item) != '']
					item_dict = {'title':title,'list':ent_list}
					context.append(json.dumps(item_dict))

		if context != []:
			with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
				f.write('\n'.join(context))
			f.close()

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
	parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
	
	args = parser.parse_args()

	dump_dir = os.listdir(args.input_dir)
	tasks = list(split(dump_dir, args.num_process))

	processes = [mp.Process(target=merge_task, args=(tasks[i], args)) for i in range(args.num_process)]

	for p in processes:
		p.start()

	for p in processes:
		p.join()

if __name__ == '__main__':
	main()

import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy

"""
parse xml format and clean text 

"""

def merge_task(task_list, invalid, args):
	nlp = spacy.load("en_core_web_lg", disable=['ner'])
	for folder in task_list:
		outputname = 'JSON_{}'.format(folder)
		working_dir = '{}/{}'.format(args.input_dir,folder)
		context = []
		for fname in os.listdir(working_dir):
			with open('{}/{}'.format(working_dir,fname), 'r') as f:
				raw = f.read()
			f.close()
			soup = BeautifulSoup(raw,'lxml')
			docs = soup.find_all('doc')
			for doc in tqdm(docs, desc='{}'.format(fname), mininterval=30):
				title = doc['title']
				if title in invalid:
					continue # filter out invalid documents
				else:
					_ = [s.extract() for s in doc('ref')]
					did = doc['id']
					text = parse(doc.text)
					paragraphs = [line for line in text.splitlines() if line != title and len(line.split()) >= 4]
					pid = 0
					for para in paragraphs:
						sid = 0
						doc = nlp(para)
						for sent in doc.sents:
							nsubj = []
							for chunk in sent.noun_chunks:
							    if chunk.root.dep_ in ['nsubjpass', 'nsubj']:
							        nsubj += [chunk.text.lower()]
							context.append(json.dumps({'title':title.lower(), 'did':did, 'pid':pid, 'sid':sid, 'text':sent.text.lower(), 'nsubj':nsubj}))
							sid += 1
						pid +=1

		if context != []:
			with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
				f.write('\n'.join(context))
			f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def parse(text):
    new_text = text.replace('<br>','\n')
    new_text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'', new_text)
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
    new_text = new_text.replace('(', '').replace(')','').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    new_text = new_text.encode("ascii", errors="ignore").decode()
    return new_text.strip()

def main():
	parser = argparse.ArgumentParser(description="parse xml format and clean text")
	parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
	parser.add_argument('--invalid_list', type=str, default='', help='list of invalid pages')
	
	args = parser.parse_args()

	with open(args.invalid_list, 'r') as f:
		invalid = f.read()
	f.close()

	invalid = set(invalid.split('\n'))

	dump_dir = os.listdir(args.input_dir)
	tasks = list(split(dump_dir, args.num_process))

	processes = [mp.Process(target=merge_task, args=(tasks[i], invalid, args)) for i in range(args.num_process)]

	for p in processes:
		p.start()

	for p in processes:
		p.join()

if __name__ == '__main__':
	main()

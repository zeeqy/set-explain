import json, sys, os, re
import argparse
import bisect
import spacy
from tqdm import tqdm
nlp = spacy.load("en_core_web_lg", disable=['ner'])
import threading

"""
add special POS to sentence level json

"""

def merge_task(task_list, args):
	for fname in task_list:
		outputname = 'SPECIAL_SENTENCE_{}'.format(fname.split('_')[-1])
		context = []

		with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
			doc = f.readlines()
		f.close()

		for item in tqdm(doc, desc='{}'.format(fname), miniters=1000):
			item_dict = json.loads(item)
			doc = nlp(item_dict['text'])
			item_dict['nsubj'] = [{'npsubj':chunk.text, 'nproot':chunk.root.text} for chunk in doc.noun_chunks if chunk.root.dep_ == 'nsubj']
			context.append(json.dumps(item_dict))
		
		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join(context))
		f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
	parser = argparse.ArgumentParser(description="Break document level json")
	parser.add_argument('--input_dir', type=str, default='', help='json document directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
	
	args = parser.parse_args()

	input_dir = os.listdir(args.input_dir)
	tasks = list(split(input_dir, args.num_process))

	threads = []
	for i in range(args.num_process):
		t = threading.Thread(target=merge_task, args=(tasks[i], args, ))
		threads.append(t)
		t.start()

if __name__ == '__main__':
	main()
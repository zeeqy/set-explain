import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from tqdm import tqdm

"""
break document level json to paragraph level json

"""

def merge_task(task_list, args):
	for fname in task_list:
		outputname = 'PARAGRAPH_{}'.format(fname.split('_')[-1])
		context = []

		with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
			doc = f.readlines()
		f.close()

		for item in tqdm(doc, desc='{}'.format(fname), mininterval=30):
			item_dict = json.loads(item)
			title = item_dict['title']
			para_text = item_dict['text'].split('\n')
			pid = 0
			for p in para_text:
				para_json = {}
				para_json['title'] = title
				para_json['did'] = item_dict['id']
				para_json['pid'] = pid
				para_json['text'] = p.lower()
				pid += 1
				context.append(json.dumps(para_json))
		
		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join(context))
		f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
	parser = argparse.ArgumentParser(description="Keep json format and clean text")
	parser.add_argument('--input_dir', type=str, default='', help='json document directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
	
	args = parser.parse_args()

	input_dir = os.listdir(args.input_dir)
	tasks = list(split(input_dir, args.num_process))

	processes = [mp.Process(target=merge_task, args=(tasks[i], args)) for i in range(args.num_process)]

	for p in processes:
		p.start()

	for p in processes:
		p.join()

if __name__ == '__main__':
	main()
	
import json, sys, os
import argparse
import threading

"""
Collect wiki document length

"""

def merge_task(task_list, args):
	for folder in task_list:
		stats = []
		outputname = 'LENGTH_{}'.format(folder) 
		working_dir = '{}/{}'.format(args.input_dir,folder)
		for fname in os.listdir(working_dir):
			with open('{}/{}'.format(working_dir,fname), 'r') as f:
				raw = f.readlines()
			f.close()
			for item in raw:
				item_dict = json.loads(item)
				title = item_dict['title']
				length = len(item_dict['text'])
				stats.apped((title, length))
		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join('{} {}'.format(x[0],x[1]) for x in stats))
		f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
	parser = argparse.ArgumentParser(description="Merge json in to corpus")
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
	
import json, sys, os
import argparse
import threading

def merge_task(task_list, args):
	for folder in task_list:
		context = []
		outputname = 'CONCAT_{}'.format(folder) 
		working_dir = '{}/{}'.format(args.input_dir,folder)
		for fname in os.listdir(working_dir):
			with open('{}/{}'.format(working_dir,fname), 'r') as f:
				raw = f.readlines()
			f.close()
			for item in raw:
				item_dict = json.loads(item)
				paragraph = parse(item_dict['text'])
				context.append(paragraph)
		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join(context))
		f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def parse(text):
	sentences = text.split('\n')
	#remove invalid sentences
	sentences = [line for line in sentences if len(line) >= 20]
	return '\n'.join(sentences)

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
	
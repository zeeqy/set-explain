import json, sys, os, re, requests, time
import argparse
import bisect
import nltk
import threading
import subprocess
from tqdm import tqdm

URL = 'http://localhost:9071/?properties={"annotators": "parse", "outputFormat": "json"}'

"""
add nsubj tag to sentence level json

"""
def extract_np(psent):
	try:
		psent.label()
	except AttributeError:
		return []
	
	np_list = []
	if psent.label() == 'NP':
		np_str = ' '.join(psent.leaves()).replace('``','"').replace("''",'"')
		np_str = re.sub(r'\s+([?.!,:; @+-=<>{}#$%^&*()]|(\'s))', r'\1', np_str)
		doubleq = re.findall(r'\"(.+?)\"',np_str)
		for st in doubleq:
			np_str = np_str.replace(st, st.strip())
		return np_str
	else:
		for child in psent:
			rec = extract_np(child)
			if rec != []:
				if isinstance(rec, list):
					np_list += rec
				else:
					np_list.append(rec)
		return np_list

def findNPRange(nps, text):
	nprange = []
	for nphrase in nps:
		start_idx = text.find(nphrase)
		if start_idx == -1:
			continue
		end_idx = start_idx + len(nphrase)
		nprange.append([nphrase, start_idx, end_idx])
	return nprange

def findNsubj(basicDep, tokens):
	nsubj = []
	for parsed in [depparsed for depparsed in basicDep if depparsed['dep'] == 'nsubj']:
		text = parsed['dependentGloss']
		index = basicDep.index(parsed) - 1
		token = tokens[index]
		nsubj.append([text,token['characterOffsetBegin'], token['characterOffsetEnd']])
	return nsubj


def merge_task(task_list, args):
	for fname in task_list:
		outputname = 'SPECIAL_SENTENCE_{}'.format(fname.split('_')[-1])
		context = []

		with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
			doc = f.readlines()
		f.close()

		#for item in doc:
		for i in tqdm(range(len(doc)), desc='{}'.format(fname)):
			item = doc[i]
			item_dict = json.loads(item)
			text = re.sub(r'\s+([?.!,:; @+-=<>{}#$%^&*()]|(\'s))', r'\1', item_dict['text'])
			item_dict['text'] = text
			r = requests.post(URL, data=text)
			try:
				content = r.json()
			except json.decoder.JSONDecodeError:
				continue
			tree = nltk.tree.ParentedTree.fromstring(content['sentences'][0]['parse'])
			
			nps = extract_np(tree)
			nsubj = findNsubj(content['sentences'][0]['basicDependencies'], content['sentences'][0]['tokens'])
			nprange = findNPRange(nps, text)
			
			if len(nsubj) == 0:
				continue
			
			np_subj = []
			for ns in nsubj:
				np_subj += [npr[0] for npr in nprange if npr[2] >= ns[2] and npr[1] <= ns[1]]

			item_dict['npsubj'] = np_subj

			context.append(item_dict)

		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join(context))
		f.close()

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
	print("Main function start")
	sys.stdout.flush()
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
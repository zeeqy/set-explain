import json, sys, os, re
import argparse
import threading

def merge_task(task_list, args):
	with open('{}/wiki_entities.json'.format(args.input_json), 'r') as f:
		entity_dict = json.loads(f.read())
	f.close()
	entity_list = list(entity_dict.keys())

	for folder in task_list:
		sentences = []
		outputname = 'SENTENCE_INDEX_{}'.format(folder) 
		working_dir = '{}/{}'.format(args.input_dir,folder)
		for fname in os.listdir(working_dir):
			with open('{}/{}'.format(working_dir,fname), 'r') as f:
				raw = f.readlines()
			f.close()

			for item in raw:
				item_dict = json.loads(item)
				title = re.sub(r'\([^)]*\)', '', item_dict['title']).lower().strip()
				sentences += parse(item_dict['text'], title, entity_dict, entity_list)

		with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join(sentences))
		f.close()

def entity_match(sentence, entity_list):
	entities = []
	entities = list(filter(lambda e: e in sentence, entity_list))
	return entities

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def parse(text, title, entity_dict, entity_list):
	new_text = text.replace('<br>','\n')
	new_text = new_text.replace('\n\n',' ')
	new_text = new_text.replace('\n',' ')
	new_text = new_text.replace('  ',' ')
	new_text = new_text.replace('<nowiki>','')
	new_text = new_text.replace('</nowiki>','')
	new_text = new_text.replace('()','')
	sentences = new_text.split('. ')
	sentence_list = []
	for s in sentences:
		if len(s) >= 20:
			sentence_dict = {'sentence':s, 'title':title, 'entities':[]}
			entities = entity_match(s, entity_list)
			for e in entities:
				sentence_dict['entities'].append(entity_dict[e])
			sentence_list.append(sentence_dict)
	return sentence_list

def main():
	parser = argparse.ArgumentParser(description="Merge json in to corpus")
	parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--input_json', type=str, default='', help='entity json directory')
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
	
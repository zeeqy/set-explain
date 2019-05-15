# -*- coding: utf-8 -*-
import json, sys, os
import threading
import queue

class matching_tools(object):
	def __init__(self, entity_dir, inverted_dir, sentence_dir, num_process):
		self.entity_dir = entity_dir
		self.inverted_dir = inverted_dir
		self.sentence_dir = sentence_dir
		self.num_process = num_process

		with open('{}/INVERTED_INDEX.json'.format(self.inverted_dir), 'r') as f:
			raw_index = f.read()
		f.close()
		self.inverted_index = json.loads(raw_index)

		with open('{}/entity2id.txt'.format(self.entity_dir), 'r') as f:
			raw_entity2id = f.read()
		f.close()

		self.entity2id = json.loads(raw_entity2id)

		with open('{}/id2entity.txt'.format(self.entity_dir), 'r') as f:
			raw_id2entity = f.read()
		f.close()

		self.id2entity = json.loads(raw_id2entity)

		with open('{}/entitylist.txt'.format(self.entity_dir), 'r') as f:
			raw_list = f.read()
		f.close()

		self.entityset = set(raw_list.split('\n'))

	def validEntity(self, entity):
		return True if entity in self.entityset else False

	def eidFinder(self, entity):
		return self.entity2id[entity]

	def entityMentioned(self, entity):
		if self.validEntity(entity):
			eid = '{}'.format(self.eidFinder(entity))
			return self.inverted_index[eid]
		else:
			return False

	def _split(self, a, n):
		k, m = divmod(len(a), n)
		return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

	def storeInQueue(self, f):
		def wrapper(*args):
			my_queue.put(f(*args))
		return wrapper

	@storeInQueue
	def merge_task(self, task_list, mentionedSet):
		content = []
		for fname in task_list:

			sent_json = {}
			
			with open('{}/{}'.format(self.sentence_dir, fname), 'r') as f:
				for line in f:
					sent_json.update(json.loads(line))
			f.close()

			key_set = set(sent_json.keys())
			inter = mentionedSet.intersection(key_set)

			for key in inter:
				content.append(json.dumps({key:sent_json[key]}))
		return content

	def key2Text(self, mentionedKeys):
		task_list = os.listdir(self.sentence_dir)

		mentionedSet = set(mentionedKeys)

		content = []

		tasks = list(self._split(task_list, self.num_process))

		sent_queue = queue.Queue()

		for i in range(self.num_process):
			t = threading.Thread(target=self.merge_task, args=(tasks[i], mentionedSet, ))
			t.start()
			content += sent_queue.get()

		return content
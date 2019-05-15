# -*- coding: utf-8 -*-
import json, sys, os

class matching_tools(object):
	def __init__(self, entity_dir, inverted_dir, sentence_dir):
		self.entity_dir = entity_dir
		self.inverted_dir = inverted_dir
		self.sentence_dir = sentence_dir
		self.load_sent = False
		self.sent_json = {}

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

	def loadSent(self):
		if self.load_sent == False:
			with open('{}/SENTENCE_ENTITY.txt'.format(self.sentence_dir), 'r') as f:
				for line in f:
					self.sent_json.update(json.loads(line))
			f.close()

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

	def key2Text(self, mentionedKeys):
		mentionedSet = set(mentionedKeys)
		content = []

		for key in mentionedSet:
			content.append(json.dumps({key:self.sent_json[key]}))

		return content
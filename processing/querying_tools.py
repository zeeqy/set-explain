# -*- coding: utf-8 -*-
import json, sys, os
import pandas as pd

class matching_tools(object):
    def __init__(self, entity_dir, inverted_dir, sentence_dir):
        self.entity_dir = entity_dir
        self.inverted_dir = inverted_dir
        self.sentence_dir = sentence_dir
        self.load_sent = False

        self.inverted_index = pd.read_csv('{}/inverted_index.csv'.format(self.inverted_dir),
                                delimiter='\t', error_bad_lines=False,
                                header=None, names=['eid','mid'], dtype={'eid':str,'eid':str})

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

            self.sent_index = pd.read_csv('{}/sentence_entity.csv'.format(self.sentence_dir),
                                delimiter='|', error_bad_lines=False,
                                header=None, names=['mid','title','did', 'pid', 'sid', 'mentioned', 'text'],
                                dtype={'mid':str,'title':str,'did':str, 'pid':str, 'sid':str, 'mentioned':str, 'text':str})
            self.load_sent = True

    def validEntity(self, entity):
        return True if entity in self.entityset else False

    def eidFinder(self, entity):
        return self.entity2id[entity]

    def entityMentioned(self, entity):
        if self.validEntity(entity):
            eid = '{}'.format(self.eidFinder(entity))
            return self.inverted_index.loc[self.inverted_index['eid'] == eid].mid.tolist()[0].split(',')
        else:
            return False

    def key2Text(self, mentionedKeys):
        mentionedSet = set(mentionedKeys)
        return self.sent_index.loc[self.sent_index.mid.isin(mentionedSet)]

    def parseMentions(self, mlist):
        mentions = {}
        for rec in mlist:
            part = rec.split('-')
            if part[0] not in mentions.keys():
                mentions.update({part[0]:{part[1]:set(part[2])}})
            elif part[1] not in mentions[part[0]].keys():
                mentions[part[0]].update({part[1]:set(part[2])})
            else:
                mentions[part[0]][part[1]].add(part[2])
        return mentions
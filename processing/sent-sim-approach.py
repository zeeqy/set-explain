#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import spacy
import textacy
import numpy as np
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
import nltk
import phrasemachine
from scipy import spatial
import time
from collections import Counter
from scipy.stats.mstats import gmean, hmean
from scipy.stats import skew
stop = set(stopwords.words('english'))

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def sent_search(params):
    (task_list, query, input_dir) = params

    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    freq = dict()

    for ent in query:
        freq.update({ent:{'total':0}})

    context = dict((ent,[]) for ent in query)

    for fname in task_list:

        with open('{}/{}'.format(input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in tqdm(doc, desc='{}'.format(fname), mininterval=10):
            try:
                item_dict = json.loads(item)
            except:
                print(fname, item)
                sys.stdout.flush()
                continue
            
            if len(item_dict['text'].split()) > 30:
                continue

            entity_text = set([em for em in item_dict['entityMentioned']])

            for ent in query:
                if ent not in entity_text:
                    continue
                else:
                    #doc = nlp(item_dict['text'])
                    #unigram = [token.lemma_ for token in textacy.extract.ngrams(doc,n=1, filter_nums=True, filter_punct=True, filter_stops=True)]
                    #item_dict['unigram'] = unigram
                    context[ent].append(item_dict)

                    freq[ent]['total'] += 1
                    if item_dict['did'] in freq[ent]:
                        freq[ent][item_dict['did']] += 1
                    else:
                        freq[ent].update({item_dict['did']:1})
    
    return {'context':context, 'freq':freq}


# In[2]:


#def main_thrd(query, num_process, input_dir, target):
query = ['kingston', 'ottawa', 'london']
target = 'cities in ontario'
input_dir = '/mnt/nfs/work1/allan/zhiqihuang/wiki_preprocessing/wikixmlentity_10'
num_process = 10

start_time = time.time()
nlp = spacy.load('en_core_web_lg', disable=['ner'])

##### sentence search #####
input_files = os.listdir(input_dir)
tasks = list(split(input_files, num_process))

inputs = [(tasks[i], query, input_dir) for i in range(num_process)]

with Pool(num_process) as p:
    search_results = p.map(sent_search, inputs)

search_merge = search_results[0]['context']
count_merge = search_results[0]['freq']

for pid in range(1, len(search_results)):
    tmp_context = search_results[pid]['context']
    tmp_freq = search_results[pid]['freq']
    for ent in query:
        search_merge[ent] += tmp_context[ent]
        count_merge[ent]['total'] += tmp_freq[ent]['total']
        tmp_freq[ent].pop('total', None)
        count_merge[ent].update(tmp_freq[ent])

for ent in query:
    for index in range(len(search_merge[ent])):
        search_merge[ent][index]['doc_score'] = count_merge[ent][search_merge[ent][index]['did']]/count_merge[ent]['total']

print("--- search use %s seconds ---" % (time.time() - start_time))
sys.stdout.flush()


# In[3]:


cooccur_ent = set([ent for sent in search_merge[query[0]] for ent in sent['entityMentioned']])
for ent in query:
    cooccur_ent = cooccur_ent.intersection(set([ent for sent in search_merge[ent] for ent in sent['entityMentioned']]))
for ent in query:
    cooccur_ent.discard(ent)
print('cooccur entities: ', len(cooccur_ent))


# In[4]:


# def jaccard_similarity(list1, list2):
#     s1 = set(list1)
#     s2 = set(list2)
#     return len(s1.intersection(s2)) / len(s1.union(s2))

entity_sents = {}
for ent in query:
    entity_sents.update({ent:{}})  
    for sent in search_merge[ent]:
            for item in cooccur_ent.intersection(set(sent['entityMentioned'])):
                if item in entity_sents[ent].keys():
                    entity_sents[ent][item].append(sent)
                else:
                    entity_sents[ent].update({item:[sent]})


# In[5]:


import nltk
import copy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
STOP = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()


# In[6]:


entity_sents_clean = {}
for ent in query:
    entity_sents_clean.update({ent:{}})
    for item, sents in entity_sents[ent].items():
        doc_weight = {}
        for sent in entity_sents[ent][item]:
            doc_weight[sent['did']] = sent['doc_score']
        topdid = [doc[0] for doc in sorted(doc_weight.items(), key=lambda x: x[1], reverse=True)[:5]]
        entity_sents_clean[ent].update({item:[]})  
        for sent in sents:
            if sent['did'] in topdid:
                tokens = nltk.word_tokenize(sent['text'])
                clean_punt = [t.lower() for t in tokens if t.isalpha()]
                clean_stopwords = [t for t in clean_punt if t not in STOP]
                clean_tokens = [LEMMA.lemmatize(t) for t in clean_stopwords]
                if clean_tokens != []:
                    copy_sent = copy.deepcopy(sent)
                    copy_sent['text_clean'] = clean_tokens #' '.join(clean_tokens)
                    entity_sents_clean[ent][item].append(copy_sent)


# In[ ]:


key = 'canada'

from gensim.models import Word2Vec
from fse.models import Average
from fse import IndexedList

sentences = []
index_list = []
for ent in query:
    sentences += [sent['text_clean'] for sent in entity_sents_clean[ent][key]]
    index_list.append(range(len(entity_sents_clean[ent][key])))

ft = Word2Vec(sentences, min_count=3)
model = Average(ft)
model.train(IndexedList(sentences))

prod = list(product(*index_list))
best_sim = 0
for pair in tqdm(prod, desc='wmd-{}'.format(key), mininterval=10):
    i = pair[0]
    j = pair[1] + len(index_list[0])
    k = pair[2] + len(index_list[0]) + len(index_list[1])
    sim = gmean([model.sv.similarity(i,j), model.sv.similarity(i,k), model.sv.similarity(j,k)])
    if sim > best_sim:
        best_sim = sim
        best_sent = pair
        concat_index = [i,j,k]

print('best sent pair:', best_sent)
print('best sim:', best_sim)
print('concat list text tokens:')
for index in concat_index:
    print(sentences[index])

print('verify original tokens:')
for index in range(len(query)):
    print(entity_sents_clean[query[index]][key][best_sent[index]]['text_clean'])

print('full sentences:')
for index in range(len(query)):
    print(entity_sents_clean[query[index]][key][best_sent[index]]['text'])

# best_sim = 0
# for sent1 in entity_sents_clean['kingston']['canada']:
#     for sent2 in entity_sents_clean['ottawa']['canada']:
#         for sent3 in entity_sents_clean['london']['canada']:
#             if sent1['text_clean'] == sent2['text_clean'] or sent1['text_clean'] == sent3['text_clean'] or sent2['text_clean'] == sent3['text_clean']:
#                 continue
#             sim = gmean([jaccard_similarity(sent1['text_clean'], sent2['text_clean']),jaccard_similarity(sent2['text_clean'], sent3['text_clean']),jaccard_similarity(sent1['text_clean'], sent3['text_clean'])])
#             if sim > best_sim:
#                 best_sim = sim
#                 best_sent = (sent1, sent2, sent3)


# In[ ]:


# from gensim.models import FastText
# sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
# ft = FastText(sentences, min_count=1, size=10)

# from fse.models import Average
# from fse import IndexedList
# model = Average(ft)
# model.train(IndexedList(sentences))

# model.sv.similarity(0,1)


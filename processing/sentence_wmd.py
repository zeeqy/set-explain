import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
import collections
from tqdm import tqdm
from random import shuffle
import copy
import spacy
import wmd
import numpy as np
from itertools import product
import gc

"""
search sentence based on wmd

"""

def merge_task(params):
    (task_list, args, queries_dict) = params

    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    context = copy.deepcopy(queries_dict)
    
    for index in range(len(context)):
        query = context[index]
        for ent in query['entities']:
            query.update({ent:[]})
        context[index] = query

    for fname in task_list:

        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in tqdm(doc, desc='{}'.format(fname), mininterval=30):
            try:
                item_dict = json.loads(item)
            except:
                print(fname, item)
                sys.stdout.flush()
                continue

            if item_dict['pid'] != 0 or item_dict['sid'] != 0:
                continue

            entity_text = set([em for em in item_dict['entityMentioned']])
            for index in range(len(context)):
                query = context[index]
                
                keys = set(query['keywords'])
                if keys.intersection(entity_text) == set():
                    continue
                
                for ent in query['entities']:
                    if ent not in entity_text:
                        continue
                    doc = nlp(item_dict['text'])
                    nsubj = [{'npsubj':chunk.text, 'nproot':chunk.root.text} for chunk in doc.noun_chunks if chunk.root.dep_ in ['nsubjpass', 'nsubj']]
                    for ns in nsubj:
                        if ent == ns['nproot'] or ent == ns['npsubj']:
                            context[index][ent].append(item_dict)
    del nlp
    gc.collect()
    return context

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def merge_wmd(params):
    nlp = spacy.load('en_core_web_lg', disable=['ner'])
    nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
    context = []
    for param in params:
        (qid, sents) = param
        filtered = [s for s in sents if s != []]
        index_list = [range(len(s)) for s in filtered]
        best_wmd = 1e6
        best_pair = []

    #     if len(filtered) == 3:
    #         # first layer
    #         for i in tqdm(index_list[0], desc='wmd-{}-3layer'.format(qid), mininterval=30):
    #             # last layer
    #             for j in index_list[1]:
    #                 # suppose 3 layer
    #                 for k in index_list[2]:
    #                     doc1 = nlp(filtered[0][i]['text'])
    #                     doc2 = nlp(filtered[1][j]['text'])
    #                     doc3 = nlp(filtered[2][k]['text'])
    #                     dist = doc1.similarity(doc2) + doc2.similarity(doc3)
    #                     if dist < best_wmd:
    #                         best_wmd = dist
    #                         best_pair = [filtered[0][i],filtered[1][j],filtered[2][k]]
        
    #     elif len(filtered) == 2:
    #         # first layer
    #         for i in tqdm(index_list[0], desc='wmd-{}-2layer'.format(qid), mininterval=30):
    #             # last layer
    #             for j in index_list[1]:
    #                 doc1 = nlp(filtered[0][i]['text'])
    #                 doc2 = nlp(filtered[1][j]['text'])
    #                 dist = doc1.similarity(doc2)
    #                 if dist < best_wmd:
    #                     dist = best_wmd
    #                     best_pair = [filtered[0][i],filtered[1][j]]

    #     elif len(filtered) == 1:
    #         best_pair = [filtered[0][0]]

    #     context.append([qid, best_pair])
    # return context

        prod = list(product(*index_list))
        
        if len(prod) > 500000 or len(filtered) < 2:
            continue
        
        for pair in tqdm(prod, desc='wmd-{}'.format(qid), mininterval=30):
            sents_pair = [filtered[index][pair[index]] for index in range(len(pair))]
            current_wmd = 0
            for index in range(len(sents_pair)-1):
                doc1 = nlp(sents_pair[index]['text'])
                doc2 = nlp(sents_pair[index+1]['text'])
                current_wmd += doc1.similarity(doc2)

            if current_wmd < best_wmd:
                best_wmd = current_wmd
                best_pair = sents_pair

        context.append([qid, best_pair])
    del nlp
    gc.collect()  
    return context


def main():
    parser = argparse.ArgumentParser(description="search sentence based on wmd")
    parser.add_argument('--input_dir', type=str, default='', help='autophrase parsed directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    parser.add_argument('--output_prefix', type=str, default='', help='output filename')
    parser.add_argument('--query_file', type=str, default='', help='search query')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    
    args = parser.parse_args()
    
    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    with open('{}'.format(args.query_file), 'r') as f:
        doc = f.readlines()
    f.close()

    queries_dict = []
    for item in doc:
        queries_dict.append(json.loads(item))

    inputs = [(tasks[i], args, queries_dict) for i in range(args.num_process)]

    pool = Pool(args.num_process)
    search_results = pool.map(merge_task, inputs)
    pool.close()
    pool.join()
    
    merge_results = search_results[0]

    for pid in range(1, len(search_results)):
        res = search_results[pid]
        for qid in range(len(res)):
            for ent in merge_results[qid]['entities']:
                merge_results[qid][ent] += res[qid][ent]

    #wmd all sentence
    batch_combined = []
    wmd_results = []
    minibatch = 1000

    chunks = [merge_results[i * minibatch:(i + 1) * minibatch] for i in range((len(merge_results) + minibatch - 1) // minibatch )]

    for batch in chunks:
        tasks = list(split(range(len(batch)), args.num_process))
        inputs = []
        for i in range(args.num_process):
            inputs.append([(qid, [batch[qid][ent] for ent in batch[qid]['entities']])  for qid in tasks[i]])

        pool = Pool(args.num_process)
        wmd_results = pool.map(merge_wmd, inputs)
        pool.close()
        pool.join()

        wmd_results = [item for sublist in wmd_results for item in sublist]

        for res in wmd_results:
            batch[res[0]]['best_context'] = res[1]

        batch_combined += batch
        
    with open('{}/{}_full.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([json.dumps({k: v for k, v in res.items() if k in ['title', 'entities', 'best_context']}) for res in batch_combined if 'best_context' in res.keys()]))
    f.close()

    transform_res = []
    for res in batch_combined:
        if 'best_context' not in res.keys():
            continue
        target = res['title'].strip()
        context = ''
        if len(res['best_context']) != 0:
            context = ' '.join([s['text'] for s in res['best_context']])
            transform_res.append({'context': context.strip(), 'target': target})

    distinct_sets = list(set([res['target'] for res in transform_res]))

    distinct_sets = np.unique(distinct_sets).tolist()

    np.random.shuffle(distinct_sets)

    valid = set(distinct_sets[0:int(len(distinct_sets) * 0.2)])
    test = set(distinct_sets[int(len(distinct_sets) * 0.2) : 2 * int(len(distinct_sets) * 0.2)])
    train = set(distinct_sets[2 * int(len(distinct_sets) * 0.2):])

    valid_set = [res for res in transform_res if res['target'] in valid]
    test_set = [res for res in transform_res if res['target'] in test]
    train_set = [res for res in transform_res if res['target'] in train]

    with open('{}/{}_train_data.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([res['context'] for res in train_set]))
    f.close()

    with open('{}/{}_train_target.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([res['target'] for res in train_set]))
    f.close()

    with open('{}/{}_val_data.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([res['context'] for res in valid_set]))
    f.close()

    with open('{}/{}_val_target.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([res['target'] for res in valid_set]))
    f.close()

    with open('{}/{}_test_data.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([res['context'] for res in test_set]))
    f.close()

    with open('{}/{}_test_target.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([res['target'] for res in test_set]))
    f.close()

if __name__ == '__main__':
    main()
    
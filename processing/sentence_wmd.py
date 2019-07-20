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
from itertools import product

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
                for ent in query['entities']:
                    if ent not in entity_text:
                        continue
                    doc = nlp(item_dict['text'])
                    nsubj = [{'npsubj':chunk.text, 'nproot':chunk.root.text} for chunk in doc.noun_chunks if chunk.root.dep_ in ['nsubjpass', 'nsubj']]
                    for ns in nsubj:
                        if ent == ns['nproot'] or ent == ns['npsubj']:
                            context[index][ent].append(item_dict)
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

        if len(filtered) == 3:
            # first layer
            for i in tqdm(index_list[0], desc='wmd-{}-3layer'.format(qid), mininterval=30):
                # last layer
                for j in index_list[1]:
                    # suppose 3 layer
                    for k in index_list[2]:
                        doc1 = nlp(filtered[0][i]['text'])
                        doc2 = nlp(filtered[1][j]['text'])
                        doc3 = nlp(filtered[2][k]['text'])
                        dist = doc1.similarity(doc2) + doc2.similarity(doc3)
                        if dist < best_wmd:
                            best_wmd = dist
                            best_pair = [filtered[0][i],filtered[1][j],filtered[2][k]]
        
        elif len(filtered) == 2:
            # first layer
            for i in tqdm(index_list[0], desc='wmd-{}-2layer'.format(qid), mininterval=30):
                # last layer
                for j in index_list[1]:
                    doc1 = nlp(filtered[0][i]['text'])
                    doc2 = nlp(filtered[1][j]['text'])
                    dist = doc1.similarity(doc2)
                    if dist < best_wmd:
                        dist = best_wmd
                        best_pair = [filtered[0][i],filtered[1][j]]

        elif len(filtered) == 1:
            best_pair = [filtered[0][0]]

        context.append([qid, best_pair])
    return context

    # prod = list(product(*index_list))
    # best_wmd = 1e10
    # best_pair = []
    # for pair in tqdm(prod, desc='wmd-{}'.format(qid), mininterval=30):
    #     sents_pair = [filtered[index][pair[index]] for index in range(len(pair))]
    #     current_wmd = 0
    #     for index in range(len(sents_pair)-1):
    #         doc1 = nlp(sents_pair[index]['text'])
    #         doc2 = nlp(sents_pair[index+1]['text'])
    #         current_wmd += doc1.similarity(doc2)

    #     if current_wmd < best_wmd:
    #         best_wmd = current_wmd
    #         best_pair = sents_pair
    # return (qid, best_pair)


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

    with Pool(args.num_process) as p:
        search_results = p.map(merge_task, inputs)
    
    merge_results = search_results[0]

    for pid in range(1, len(search_results)):
        res = search_results[pid]
        for qid in range(len(res)):
            for ent in merge_results[qid]['entities']:
                merge_results[qid][ent] += res[qid][ent]

    #wmd all sentence
    wmd_results = []
    tasks = list(split(range(len(merge_results)), args.num_process))
    inputs = []
    for i in range(args.num_process):
        inputs.append([(qid, [merge_results[qid][ent] for ent in merge_results[qid]['entities']])  for qid in tasks[i]])

    with Pool(args.num_process) as p:
        wmd_results = p.map(merge_wmd, inputs)

    wmd_results = [item for sublist in l for item in wmd_results]

    for res in wmd_results:
        merge_results[res[0]]['best_context'] = res[1]
        
    with open('{}/{}_full.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([json.dumps({k: v for k, v in res.items() if k in ['title', 'entities', 'best_context']}) for res in merge_results if res['best_context'] != []]))
    f.close()

    transform_res = []
    for res in merge_results:
        target = res['title']
        context = ''
        if len(res['best_context']) != 0:
            context = ' '.join([s['text'] for s in res['best_context']])
            transform_res.append({'context': context.strip(), 'target': target})

    shuffle(transform_res)

    valid_set = transform_res[0:int(len(transform_res) * 0.2)]
    test_set = transform_res[int(len(transform_res) * 0.2) : 2 * int(len(transform_res) * 0.2)]
    train_set = transform_res[2 * int(len(transform_res) * 0.2):]

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
    
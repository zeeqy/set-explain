import json, sys, os, re
import argparse
import bisect
import nltk
import multiprocessing as mp
from multiprocessing import Pool
import collections
from tqdm import tqdm
from random import shuffle
import copy

"""
search sentence based on keywords

"""

def jaccard_similarity(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))

def merge_task(params):
    (task_list, args, keywords_dict) = params

    context = copy.deepcopy(keywords_dict)
    
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

            entity_text = set([em for em in item_dict['entityMentioned']])
            for index in range(len(context)):
                query = context[index]
                for ent in query['entities']:
                    cooccur = set(query['keywords'] + [ent.replace('_', ' ')])
                    if ent.replace('_', ' ') in entity_text: #cooccur.issubset(entity_text):
                        item_dict['score'] = jaccard_similarity(cooccur, entity_text)
                        context[index][ent].append(item_dict)

    return context

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
    parser = argparse.ArgumentParser(description="search sentence based on keywords")
    parser.add_argument('--input_dir', type=str, default='', help='autophrase parsed directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    parser.add_argument('--output_prefix', type=str, default='', help='output filename')
    parser.add_argument('--keywords_file', type=str, default='', help='search keywords')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    
    args = parser.parse_args()

    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    with open('{}'.format(args.keywords_file), 'r') as f:
        doc = f.readlines()
    f.close()

    keywords_dict = []
    for item in doc:
        keywords_dict.append(json.loads(item))

    inputs = [(tasks[i], args, keywords_dict) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        search_results = p.map(merge_task, inputs)
    
    merge_results = search_results[0]

    for pid in range(1, len(search_results)):
        res = search_results[pid]
        for qid in range(len(res)):
            for ent in merge_results[qid]['entities']:
                merge_results[qid][ent] += res[qid][ent]

        
    #rank sentence
    for qid in range(len(merge_results)):
        for ent in merge_results[qid]['entities']:
            sents = merge_results[qid][ent]
            count = collections.Counter([s['title'] for s in sents])
            most_common = [common[0] for common in count.most_common(2)]
            doc_sents = [s for s in sents if s['title'] in most_common]
            if len(doc_sents) > 3:
                sorted_sents = sorted(doc_sents, key = lambda s: s['score'], reverse=True) 
                merge_results[qid][ent] = sorted_sents[0:3]
            else:
                merge_results[qid][ent] = doc_sents

    with open('{}/{}_full.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([json.dumps(res) for res in merge_results]))
    f.close()

    transform_res = []
    for res in merge_results:
        target = res['title']
        context = ''
        for ent in res['entities']:
            if len(res[ent]) != 0:
                context += ' '.join([s['text'] for s in res[ent]])
                context += ' '
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
    
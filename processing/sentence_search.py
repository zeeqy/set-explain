import json, sys, os, re
import argparse
import bisect
import nltk
import multiprocessing as mp
import collections
from tqdm import tqdm
import copy

"""
search sentence based on keywords

"""

def jaccard_similarity(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))

def merge_task(task_list, args, keywords_dict, outputs):
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

    outputs.put(context)

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

    outputs = mp.Queue()

    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    with open('{}'.format(args.keywords_file), 'r') as f:
        doc = f.readlines()
    f.close()

    keywords_dict = []
    for item in doc:
        keywords_dict.append(json.loads(item))

    search_results = []
    
    processes = [mp.Process(target=merge_task, args=(tasks[i], args, keywords_dict, outputs)) for i in range(args.num_process)]

    for p in processes:
        p.start()
        search_results.append(outputs.get())
    
    for p in processes:
        p.join()


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
            #count = collections.Counter([s['title'] for s in sents])
            #most_common = count.most_common()[0][0]
            doc_sents = [s for s in sents] #if s['title'] == most_common]
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
        transform_res.append({'context': context, 'target': target})

    with open('{}/{}_pair.txt'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([json.dumps(res) for res in transform_res]))
    f.close()

if __name__ == '__main__':
    main()
    
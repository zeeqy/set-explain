import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import spacy
import numpy as np
import time

def sent_search(params):
    (task_list, query, input_dir) = params

    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    context = dict.fromkeys(query, 0)

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

            entity_text = set([em for em in item_dict['entityMentioned']])

            for ent in query:
                if ent not in entity_text:
                    continue
                else:
                    doc = nlp(item_dict['text'])
                    if len(doc) >= 30:
                        continue
                    context[ent] += 1

    return context

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main_thrd(query, num_process, input_dir):
    start_time = time.time()
    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    ##### sentence search #####
    input_files = os.listdir(input_dir)
    tasks = list(split(input_files, num_process))
    
    inputs = [(tasks[i], query, input_dir) for i in range(num_process)]

    with Pool(num_process) as p:
        search_results = p.map(sent_search, inputs)
    
    search_merge = dict.fromkeys(query, 0)
    
    for ent in query:
        for tmp_res in search_results:
            search_merge[ent] += tmp_res[ent]

    search_merge = {k: v / total for total in (sum(search_merge.values()),) for k, v in search_merge.items()}

    print("--- search use %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()
    return search_merge

def main():
    parser = argparse.ArgumentParser(description="heuristic approach")
    parser.add_argument('--input_dir', type=str, default='', help='corpus directory')
    parser.add_argument('--query_dir', type=str, default='', help='search query')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    
    args = parser.parse_args()

    with open('{}/valid_set.txt'.format(args.query_dir), 'r') as f:
        sets = f.read().split('\n')
    f.close()
    sets = [line for line in sets if line != '']

    query_set = []
    for entry in sets:
        query_set.append(json.loads(entry))

    for item in query_set:
        seeds = [w.lower().replace('-', ' ').replace('_', ' ') for w in item['entities']]
        item['prob'] = main_thrd(seeds, args.num_process, args.input_dir)
        
        with open('{}/set_prob.txt'.format(args.query_dir), 'a+') as f:
            f.write(json.dumps(item) + '\n')
        f.close()

if __name__ == '__main__':
    main()
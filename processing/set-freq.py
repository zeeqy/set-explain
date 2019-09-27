import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import copy
import time

def sent_search(params):
    (task_list, query_set, input_dir) = params

    query_set_prob = copy.deepcopy(query_set)
    for i in range(len(query_set_prob)):
        query_set_prob[i]['prob'] = dict.fromkeys(query_set_prob[i]['entities'],0)

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
            for i in range(len(query_set_prob)):
                for ent in query_set_prob[i]['entities']:
                    if ent not in entity_text:
                        continue
                    else:
                        query_set_prob[i]['prob'][ent] += 1

    return query_set_prob

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main_thrd(query_set, num_process, input_dir):
    start_time = time.time()

    query_set_prob = []
    for query in query_set:
        item['entities'] = [w.lower().replace('-', ' ').replace('_', ' ') for w in item['entities']]
        query_set_prob.append(query)

    ##### sentence search #####
    input_files = os.listdir(input_dir)
    tasks = list(split(input_files, num_process))
    
    inputs = [(tasks[i], query_set_prob, input_dir) for i in range(num_process)]

    with Pool(num_process) as p:
        search_results = p.map(sent_search, inputs)
    
    search_merge = search_results[0]
    
    for pid in range(1, len(search_results))
        for i in range(len(query_set_prob)):
            for ent in search_merge[i]['prob'].keys():
                search_merge[i]['prob'][ent] += search_results[pid][i]['prob'][ent]

    for i in range(len(search_merge)):
        search_merge[i]['prob'] = {k: v / total for total in (sum(search_merge[i]['prob'].values()),) for k, v in search_merge[i]['prob'].items()}

    with open('{}/set_prob.txt'.format(args.query_dir), 'w+') as f:
        for item in search_merge:
            f.write(json.dumps(item) + '\n')
    f.close()

    print("--- search use %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="seeds freq")
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

    main_thrd(seeds, args.num_process, args.input_dir, query_set)

if __name__ == '__main__':
    main()
import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
import collections
from tqdm import tqdm
import spacy
import wmd
from itertools import product

"""
group sentence by cooccurrence

"""

def sent_search(params):
    (task_list, args) = params

    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    query = args.query_string.split(',')

    context = dict((ent,[]) for ent in query)

    for fname in task_list:

        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
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
                    item_dict['core'] = ' '.join([token.text for token in doc if token.is_stop == False])
                    context[ent].append(item_dict)
    return context

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

# def cooccur_search(params):
#     (ent, merge_results) = params
#     context = {'cooccur':ent}
#     for k in merge_results.keys():
#         context.update({k:[[sent['text'],sent['core']] for sent in merge_results[k] if ent in set(sent['entityMentioned'])]})
#     return context

def cooccur_cluster(params):
    (inital_sents, merge_results, args, proid) = params
    query = args.query_string.split(',')
    nlp = spacy.load('en_core_web_lg', disable=['ner'])
    nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
    context = []
    for init_sent in tqdm(inital_sents, desc='main-{}'.format(proid), mininterval=30):
        init_set = set([em for em in init_sent['entityMentioned']])
        doc_init = nlp(init_sent['core'])
        sents = [merge_results[ent] for ent in query[1:]]
        index_list = [range(len(s)) for s in sents]
        best_wmd = 1e6
        best_pair = []
        prod = list(product(*index_list))
        for pair in tqdm(prod, desc='wmd-{}-{}-{}'.format(init_sent['did'], init_sent['pid'], init_sent['sid']), mininterval=10):
            sents_pair = [sents[index][pair[index]] for index in range(len(pair))]
            
            overlap = init_set
            for sent in sents_pair:
                overlap = overlap.intersection(set([em for em in sent['entityMentioned']]))
            if overlap == set():
                continue

            current_wmd = 0
            for index in range(len(sents_pair)):
                doc = nlp(sents_pair[index]['core'])
                current_wmd += doc_init.similarity(doc)
            
            if current_wmd < best_wmd:
                best_wmd = current_wmd
                best_pair = sents_pair
        
        context.append({'init_sent':init_sent, 'best_wmd':best_wmd, 'best_pair': best_pair})

    return context

def main():
    parser = argparse.ArgumentParser(description="group sentence by cooccurrence")
    parser.add_argument('--input_dir', type=str, default='', help='autophrase parsed directory')
    #parser.add_argument('--output_dir', type=str, default='', help='output directory')
    #parser.add_argument('--output_prefix', type=str, default='', help='output filename')
    parser.add_argument('--query_string', type=str, default='', help='search query')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    
    args = parser.parse_args()
    
    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    inputs = [(tasks[i], args) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        search_results = p.map(sent_search, inputs)
    
    merge_results = search_results[0]

    query = args.query_string.split(',')

    for pid in range(1, len(search_results)):
        res = search_results[pid]
        for ent in query:
            merge_results[ent] += res[ent]

    entityMentioned = {}
    for ent in query:
        tmp = set()
        for sent in merge_results[ent]:
            tmp = tmp.union(set([em for em in sent['entityMentioned']]))
        entityMentioned.update({ent:tmp})

    cooccur = entityMentioned[query[0]]
    for ent in query:
        cooccur = cooccur.intersection(entityMentioned[ent])

    for ent in query:
        merge_results[ent] = [sent for sent in merge_results[ent] if any(x in set([em for em in sent['entityMentioned']]) for x in cooccur)]

    #wmd pair
    inital_ent = query[0]
    tasks = list(split(merge_results[inital_ent], args.num_process))
    inputs = [(tasks[i], merge_results, args, i) for i in range(args.num_process)]
    
    with Pool(args.num_process) as p:
        cluster = p.map(cooccur_cluster, inputs)

    cluster = [item for sublist in cluster for item in sublist]

    sorted_cluster = sorted(cluster, key = lambda i: i['best_wmd'])

    with open('cooccur_test.txt', "w+") as f:
        f.write('\n'.join([json.dumps(res) for res in sorted_cluster[:20]]))
    f.close()

if __name__ == '__main__':
    main()
    
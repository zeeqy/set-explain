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
import phrasemachine

def count_freq(params):
    (task_list, args) = params

    query = args.query_string.split(',')

    freq = dict()

    for ent in query:
        freq.update({ent:{'total':0}})

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
                    freq[ent]['total'] += 1
                    if item_dict['did'] in freq[ent]:
                        freq[ent][item_dict['did']] += 1
                    else:
                        freq[ent].update({item_dict['did']:1})
    return freq

def sent_search(params):
    (task_list, args, count_results) = params

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
                if item_dict['did'] not in count_results[ent] or ent not in entity_text:
                    continue
                else:
                    doc = nlp(item_dict['text'])
                    nsubj = [{'npsubj':chunk.text, 'nproot':chunk.root.text} for chunk in doc.noun_chunks if chunk.root.dep_ in ['nsubjpass', 'nsubj']]
                    for ns in nsubj:
                        if ent == ns['nproot'] or ent == ns['npsubj']:
                            tokens = [token.text for token in doc]
                            pos = [token.pos_ for token in doc]
                            phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos)
                            item_dict['doc_score'] = count_results[ent][item_dict['did']]/count_results[ent]['total']
                            item_dict['phrases'] = list(phrases['counts'])
                            context[ent].append(item_dict)
                    # doc = nlp(item_dict['text'])
                    # #item_dict['core'] = ' '.join([token.text for token in doc if token.is_stop == False])
                    # context[ent].append(item_dict)
    return context

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
    parser = argparse.ArgumentParser(description="group sentence by cooccurrence")
    parser.add_argument('--input_dir', type=str, default='', help='autophrase parsed directory')
    parser.add_argument('--query_string', type=str, default='', help='search query')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    
    args = parser.parse_args()
    query = args.query_string.split(',')

    ##### count mentions in corpus #####
    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    inputs = [(tasks[i], query) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        count_results = p.map(count_freq, inputs)

    count_merge = count_results[0]
    for pid in range(1, len(count_results)):
        tmp_res = count_results[pid]
        for ent in query:
            count_results[ent]['total'] += tmp_res[ent]['total']
            tmp_res[ent].pop('total', None)
            count_results[ent].update(tmp_res[ent])

    ##### get mentioned doc #####
    # doc_mentions = []
    # for ent in query:
    #     doc_mentions.append(list(count_results[ent].keys()))
    # doc_mentions = set(doc_mentions)

    ##### sentence search #####
    inputs = [(tasks[i], args, count_results) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        search_results = p.map(sent_search, inputs)
    
    search_merge = search_results[0]

    for pid in range(1, len(search_results)):
        tmp_res = search_results[pid]
        for ent in query:
            search_merge[ent] += tmp_res[ent]

    ##### entity cooccurrence #####
    entityMentioned = {}
    for ent in query:
        entityMentioned.update({ent:{}})
        for sent in search_merge[ent]:
            for cooent in sent['entityMentioned']:
                if cooent in query:
                    continue
                elif cooent in entityMentioned[ent]:
                    entityMentioned[ent][cooent] += sent['doc_score']
                else:
                    entityMentioned[ent].update({cooent:sent['doc_score']})

    cooccur = set(entityMentioned[query[0]].keys())
    for ent in query:
        tmp_res = set(entityMentioned[ent].keys())
        cooccur = cooccur.intersection(tmp_res)

    ##### rank cooccurrence #####
    cooccur_score = {}
    for cooent in cooccur:
        cooccur_score.update({cooent:0})
        for ent in query:
            cooccur_score[cooent] += entityMentioned[ent][cooent]

    cooccur_sorted = sorted(cooccur_score.items(), key=lambda x: x[1], reverse=True)

    print(cooccur_sorted)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
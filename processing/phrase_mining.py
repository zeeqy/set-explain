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
from itertools import combinations
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
                    # #item_dict['core'] = ' '.join([token.text for token in doc if token.is_stop == False])
                    # tokens = [token.text for token in doc]
                    # pos = [token.pos_ for token in doc]
                    # phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos)
                    # item_dict['doc_score'] = count_results[ent][item_dict['did']]/count_results[ent]['total']
                    # item_dict['phrases'] = list(phrases['counts'])
                    # context[ent].append(item_dict)
    return context

def cooccur_cluster(params):
    (cooccur, entityMentioned, query) = params
    nlp = spacy.load('en_core_web_lg', disable=['ner'])
    #nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
    context = {}
    for keyent in cooccur:
        
        sentsPool = []
        for seed in query:
            sentsPool.append(entityMentioned[seed][keyent]['sents'])

        index_list = [range(len(s)) for s in sentsPool]
        best_wmd = 0
        best_pair = []
        prod = list(product(*index_list))
        if len(prod) > 1e5:
            continue
        for pair in tqdm(prod, desc='wmd-{}'.format(keyent), mininterval=10):
            sentsPair = [sentsPool[index][pair[index]] for index in range(len(pair))]

            comb = combinations(sentsPair, 2) 
            current_wmd = 0
            for group in comb:
                doc1 = nlp(group[0])
                doc2 = nlp(group[1])
                current_wmd += doc1.similarity(doc2)

            if current_wmd > best_wmd:
                best_wmd = current_wmd
                best_pair = sentsPair
        
        context.update({keyent:{'best_pair':best_pair, 'best_wmd':best_wmd}})
    
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

    print(query)
    sys.stdout.flush()

    ##### count mentions in corpus #####
    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    inputs = [(tasks[i], args) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        count_results = p.map(count_freq, inputs)

    count_merge = count_results[0]
    for pid in range(1, len(count_results)):
        tmp_res = count_results[pid]
        for ent in query:
            count_merge[ent]['total'] += tmp_res[ent]['total']
            tmp_res[ent].pop('total', None)
            count_merge[ent].update(tmp_res[ent])

    print(count_merge)

    ##### get mentioned doc #####
    # doc_mentions = []
    # for ent in query:
    #     doc_mentions.append(list(count_results[ent].keys()))
    # doc_mentions = set(doc_mentions)

    ##### sentence search #####
    inputs = [(tasks[i], args, count_merge) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        search_results = p.map(sent_search, inputs)
    
    search_merge = search_results[0]

    for pid in range(1, len(search_results)):
        tmp_res = search_results[pid]
        for ent in query:
            search_merge[ent] += tmp_res[ent]

    for ent in query:
        for sent in search_merge[ent]:
            print(sent)
    sys.stdout.flush()

    ##### entity cooccurrence #####
    entityMentioned = {}
    for ent in query:
        entityMentioned.update({ent:{}})
        for sent in search_merge[ent]:
            for cooent in sent['entityMentioned']:
                if cooent in query:
                    continue
                elif cooent in entityMentioned[ent]:
                    entityMentioned[ent][cooent]['score'] += sent['doc_score']
                    entityMentioned[ent][cooent]['sents'].append(sent['text'])
                else:
                    entityMentioned[ent].update({cooent:{'score':sent['doc_score'], 'sents':[sent['text']]}})

    cooccur = set(entityMentioned[query[0]].keys())
    for ent in query:
        tmp_res = set(entityMentioned[ent].keys())
        cooccur = cooccur.intersection(tmp_res)

    ##### rank cooccurrence #####
    cooccur_score = {}
    for cooent in cooccur:
        cooccur_score.update({cooent:1})
        for ent in query:
            cooccur_score[cooent] += entityMentioned[ent][cooent]['score']

    cooccur_sorted = sorted(cooccur_score.items(), key=lambda x: x[1], reverse=True)

    for item in cooccur_sorted:
        print(item)
    sys.stdout.flush()

    threshold = int(0.3 * len(cooccur))

    cooccur_subset = [item[0] for item in cooccur_sorted[:threshold]]

    ##### wmd based on cooccurrence #####
    tasks = list(split(list(cooccur), args.num_process))
    inputs = [(tasks[i], entityMentioned, query) for i in range(args.num_process)]
    
    with Pool(args.num_process) as p:
        wmd_results = p.map(cooccur_cluster, inputs)

    wmd_merge = wmd_results[0]
    for pid in range(1, len(wmd_results)):
        tmp_res = wmd_results[pid]
        wmd_merge.update(tmp_res)

    sorted_wmd = sorted(wmd_merge.items(), key=lambda x : x[1]['best_wmd'])

    for item in sorted_wmd:
        print(item)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
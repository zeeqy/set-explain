import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
import collections
from tqdm import tqdm
import textacy
from textacy import similarity
import spacy
import wmd
from itertools import product
from itertools import combinations
from nltk.tokenize import MWETokenizer
import nltk
import phrasemachine
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

def sent_search(params):
    (task_list, args) = params

    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    query = args.query_string.split(',')

    freq = dict()

    for ent in query:
        freq.update({ent:{'total':0}})

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
                    if len(doc) >= 40:
                        continue

                    # if ent in item_dict['nsubj']:
                    #     tokens = [token.text for token in doc]
                    #     pos = [token.pos_ for token in doc]
                    #     phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos)
                    #     item_dict['phrases'] = list(phrases['counts'])
                    #     context[ent].append(item_dict)

                    #     freq[ent]['total'] += 1
                    #     if item_dict['did'] in freq[ent]:
                    #         freq[ent][item_dict['did']] += 1
                    #     else:
                    #         freq[ent].update({item_dict['did']:1})
                    
                    tokens = [token.text for token in doc]
                    pos = [token.pos_ for token in doc]
                    phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos)
                    item_dict['phrases'] = list(phrases['counts'])
                    context[ent].append(item_dict)

                    freq[ent]['total'] += 1
                    if item_dict['did'] in freq[ent]:
                        freq[ent][item_dict['did']] += 1
                    else:
                        freq[ent].update({item_dict['did']:1})
    
    return {'context':context, 'freq':freq}

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
        best_sim = 0
        best_pair = []
        prod = list(product(*index_list))
        if len(prod) > 1e5:
            continue
        for pair in tqdm(prod, desc='similarity-{}'.format(keyent), mininterval=10):
            sentsPair = [sentsPool[index][pair[index]]['text'] for index in range(len(pair))]

            comb = combinations(sentsPair, 2) 
            current_sim = 0
            for group in comb:
                # current_sim += similarity.jaccard(group[0],group[1])
                doc1 = nlp(group[0])
                doc2 = nlp(group[1])
                current_sim += textacy.similarity.word2vec(doc1,doc2)

            if current_sim > best_sim:
                best_sim = current_sim
                best_pair = sentsPair
        
        context.update({keyent:{'best_pair':best_pair, 'best_sim':best_sim}})
    
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
    nlp = spacy.load('en_core_web_lg')

    print(query)
    sys.stdout.flush()

    ##### sentence search #####
    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))
    
    inputs = [(tasks[i], args) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
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

    ##### entity cooccurrence #####
    entityMentioned = {}
    count_entity = {}
    for ent in query:
        entityMentioned.update({ent:{}})
        count_entity.update({ent:0})
        for sent in search_merge[ent]:
            for cooent in sent['entityMentioned']:
                if cooent in query:
                    continue
                elif cooent in entityMentioned[ent]:
                    entityMentioned[ent][cooent]['doc_score'] += sent['doc_score']
                    entityMentioned[ent][cooent]['total'] += 1
                    entityMentioned[ent][cooent]['sents'].append(sent)

                else:
                    entityMentioned[ent].update({cooent:{'total':1, 'sents': [sent], 'doc_score': sent['doc_score']}})
                count_entity[ent] += 1
                
    for ent in query:
        total = count_entity[ent]
        for cooent, value in entityMentioned[ent].items():
            entityMentioned[ent][cooent]['score'] = entityMentioned[ent][cooent]['total'] / total

    cooccur_list = {}
    for ent in query:
        cooccur_list.update({ent:set()})
        for cooent, value in entityMentioned[ent].items():
            #if entityMentioned[ent][cooent]['total'] >= 5:
            cooccur_list[ent].add(cooent)

    cooccur = cooccur_list[query[0]]
    for ent in query:
        cooccur = cooccur.intersection(cooccur_list[ent])

    ##### rank cooccurrence #####
    cooccur_score = {}
    for cooent in cooccur:
        cooccur_score.update({cooent:0})
        for ent in query:
            cooccur_score[cooent] += entityMentioned[ent][cooent]['doc_score']

    cooccur_sorted = sorted(cooccur_score.items(), key=lambda x: x[1], reverse=True)

    print(cooccur_sorted)

    sys.stdout.flush()

    fid = 1
    for ent in query:
        with open('retrieved-{}.txt'.format(fid), "w+") as f:
            for sent in search_merge[ent]:
                if set(sent['entityMentioned']).intersection(cooccur) != set():
                    f.write(json.dumps(sent) + '\n')
        f.close()
        fid += 1



    # threshold = int(0.3 * len(cooccur))

    #cooccur_subset = [item[0] for item in cooccur_sorted[:threshold]]

    ### similarity based on cooccurrence #####
    # tasks = list(split(list(cooccur), args.num_process))
    # inputs = [(tasks[i], entityMentioned, query) for i in range(args.num_process)]
    
    # with Pool(args.num_process) as p:
    #     sim_results = p.map(cooccur_cluster, inputs)

    # sim_merge = sim_results[0]
    # for pid in range(1, len(sim_results)):
    #     tmp_res = sim_results[pid]
    #     sim_merge.update(tmp_res)

    # sorted_sim = sorted(sim_merge.items(), key=lambda x : x[1]['best_sim'], reverse=True)

    # for item in sorted_sim:
    #     print(item)
    # sys.stdout.flush()

    list_phrases = []
    for ent in query:
        for sent in search_merge[ent]:
            list_phrases += sent['phrases']

    tokenizer = MWETokenizer(separator=' ')

    for e in cooccur:
        tokenizer.add_mwe(nltk.word_tokenize(e))
    
    list_phrases = set(list_phrases)
    phrases_score = {}
    for phrase in list_phrases:
        score = 0
        tokens = nltk.word_tokenize(phrase)
        nonstop_tokens = [token for token in tokens if token not in stop]
        if len(nonstop_tokens) / len(tokens) <= 0.5:
            continue
        raw_tokenized = tokenizer.tokenize(tokens)
        tokenized_set = set(raw_tokenized)
        for token in tokenized_set.intersection(cooccur):
            score += cooccur_score[token]
        phrases_score.update({phrase:score/len(nonstop_tokens)})

    phrases_sorted = sorted(phrases_score.items(), key=lambda x: x[1], reverse=True)

    print(phrases_sorted[:10])

if __name__ == '__main__':
    main()
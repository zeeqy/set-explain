import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import spacy
import textacy
import numpy as np
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
import nltk
import phrasemachine
from scipy import spatial
import time
from collections import Counter
from scipy.stats.mstats import gmean, hmean
from scipy.stats import skew
stop = set(stopwords.words('english'))

def sent_search(params):
    (task_list, query, input_dir) = params

    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    freq = dict()

    for ent in query:
        freq.update({ent:{'total':0}})

    context = dict((ent,[]) for ent in query)

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

            for ent in query:
                if ent not in entity_text:
                    continue
                else:
                    doc = nlp(item_dict['text'])
                    unigram = [token.lemma_ for token in textacy.extract.ngrams(doc,n=1, filter_nums=True, filter_punct=True, filter_stops=True)]
                    item_dict['unigram'] = unigram
                    tokens = [token.lemma_ for token in doc]
                    item_dict['tokens'] = [token.lemma_ for token in doc if not token.is_punct]
                    #item_dict['phrases'] = [chunk.lemma_ for chunk in doc.noun_chunks]
                    pos = [token.pos_ for token in doc]
                    phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos, minlen=2, maxlen=8)
                    item_dict['phrases'] = list(phrases['counts'])
                    context[ent].append(item_dict)

                    freq[ent]['total'] += 1
                    if item_dict['did'] in freq[ent]:
                        freq[ent][item_dict['did']] += 1
                    else:
                        freq[ent].update({item_dict['did']:1})
    
    return {'context':context, 'freq':freq}

def phrase_eval(params):
    list_phrases, unigram_set, target_token, idf, agg_score, pid = params

    idf_list = [*idf]
    idf_set = set(idf_list)

    tokenizer = MWETokenizer(separator=' ')
    for e in unigram_set:
        tokenizer.add_mwe(nltk.word_tokenize(e))

    phrases_score = {}
    for phrase in tqdm(list_phrases, desc='phrase-eval-{}'.format(pid), mininterval=10):
        score = 0
        tokens = nltk.word_tokenize(phrase)
        if not set(tokens).issubset(idf_set):
            continue
        nonstop_tokens = [token for token in tokens if token not in stop]
        if len(nonstop_tokens) / len(tokens) <= 0.5:
            continue
        raw_tokenized = tokenizer.tokenize(nonstop_tokens)
        tokenized_set = set(raw_tokenized)
        for token in tokenized_set.intersection(unigram_set):
            score += agg_score[token]
        score /= len(nonstop_tokens)
        

        vocab = list(set(target_token).union(set(tokens)))
        target_vec = [0] * len(vocab)
        phrase_vec = [0] * len(vocab)
        
        target_token_freq = dict(Counter(target_token))
        for token in target_token:
            index = vocab.index(token)
            target_vec[index] = target_token_freq[token]/len(target_token) * idf[token]
        
        phrase_token_freq = dict(Counter(tokens))
        for token in tokens:
            index = vocab.index(token)
            phrase_vec[index] = phrase_token_freq[token]/len(tokens) * idf[token]
        
        tfidf_sim = 1 - spatial.distance.cosine(target_vec, phrase_vec)

        phrases_score.update({phrase:{'score': score, 'eval': tfidf_sim}})

    return phrases_score

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def sublist1(lst1, lst2):
    ls1 = [element for element in lst1 if element in lst2]
    ls2 = [element for element in lst2 if element in lst1]
    return ls1 == ls2

def main_thrd(query, num_process, input_dir, target):
    start_time = time.time()
    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    ##### sentence search #####
    input_files = os.listdir(input_dir)
    tasks = list(split(input_files, num_process))
    
    inputs = [(tasks[i], query, input_dir) for i in range(num_process)]

    with Pool(num_process) as p:
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

    print("--- search use %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()

    start_time = time.time()
    unigrams = []
    for ent in query:
        for sent in search_merge[ent]:
            unigrams += sent['unigram']
    unigram_set = set(unigrams)

    N = 0
    cnt = Counter()
    for ent in query:
        N += len(search_merge[ent])
        for sent in search_merge[ent]:
            cnt.update(sent['tokens'])
    cnt = dict(cnt)

    for ent in query:
        unigram_set.discard(ent)

    idf = {}
    for key in cnt.keys():
        idf.update({key:np.log(N / cnt[key])})
    
    unigram_sents = {}
    for ent in query:
        unigram_sents.update({ent:{}})  
        for sent in search_merge[ent]:
            unigram = set(sent['unigram'])
            unigram_intersect = unigram.intersection(unigram_set)
            for item in unigram_intersect:
                if item in unigram_sents[ent].keys():
                    unigram_sents[ent][item].append(sent)
                else:
                    unigram_sents[ent].update({item:[sent]})

    score_dist = {}
    for ug in unigram_set:
        score_dist.update({ug:{}})
        for ent in query:
            score_dist[ug].update({ent:0})
            if ug in unigram_sents[ent].keys():
                did = set()
                for sent in unigram_sents[ent][ug]:
                    score_dist[ug][ent] += sent['doc_score']
                    did.add(sent['did'])

    #using rank to score unigram
    score_redist = {}
    for ent in query:
        score_redist.update({ent:dict.fromkeys(unigram_set, 0)})
        for ug in unigram_set:
            score_redist[ent][ug] = score_dist[ug][ent]    
        sorted_score = sorted(score_redist[ent].items(), key=lambda item: item[1], reverse=True)
        rank, count, previous, result = 0, 0, None, {}
        for key, num in sorted_score:
            count += 1
            if num != previous:
                rank += count
                previous = num
                count = 0
            result[key] = 1.0 / rank
        score_redist[ent] = result

    for ug in unigram_set:
        for ent in query:
            score_dist[ug][ent] = score_redist[ent][ug]

    agg_score = {}
    for ug in score_dist.keys():
        tmp_res = [item[1] for item in score_dist[ug].items()]
        agg_score.update({ug: gmean(tmp_res)})


    score_sorted = sorted(agg_score.items(), key=lambda x: x[1], reverse=True)

    print("--- unigram score %s seconds ---" % (time.time() - start_time))
    print(score_sorted[:10])
    sys.stdout.flush()
    
    start_time = time.time()

    tokenizer = MWETokenizer(separator=' ')
    for ent in query:
        tokenizer.add_mwe(nltk.word_tokenize(ent))
    
    mined_phrases = []
    query_set = set(query)
    for ent in query:
        for sent in search_merge[ent]:
            for phrase in sent['phrases']:
                tokens = nltk.word_tokenize(phrase)
                raw_tokenized = tokenizer.tokenize(tokens)
                tokenized_set = set(raw_tokenized)
                if tokenized_set.intersection(query_set) == set():
                    mined_phrases.append(phrase)

    print("--- phrase mining %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()

    start_time = time.time()

    idf_list = [*idf]
    target_doc = nlp(target)
    target_vec = [0] * len(idf_list)
    target_token = [token.lemma_ for token in target_doc if not token.is_punct]
    # target_token_freq = dict(Counter(target_token))
    # for token in target_token:
    #     index = idf_list.index(token)
    #     target_vec[index] = target_token_freq[token]/len(target_token) * idf[token]

    list_phrases = list(set(mined_phrases))

    tasks = list(split(list_phrases, num_process))

    print('target_token', target_token)
    
    inputs = [(tasks[i], unigram_set, target_token, idf, agg_score, i) for i in range(num_process)]

    phrases_score = {}
    with Pool(num_process) as p:
        eval_results = p.map(phrase_eval, inputs)

    for tmp_res in eval_results:
        phrases_score.update(tmp_res)
    
    phrases_sorted = sorted(phrases_score.items(), key=lambda x: x[1]['score'], reverse=True)

    print("--- phrase eval use %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()

    return phrases_sorted

def main():
    parser = argparse.ArgumentParser(description="heuristic approach")
    parser.add_argument('--input_dir', type=str, default='', help='corpus directory')
    parser.add_argument('--query_dir', type=str, default='', help='search query')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    parser.add_argument('--num_query', type=int, default=5, help='number of query per set')
    parser.add_argument('--query_length', type=int, default=3, help='query length')
    parser.add_argument('--entity_dir', type=str, default='', help='entity files directory')
    
    args = parser.parse_args()
    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    with open('{}/set_prob.txt'.format(args.query_dir), 'r') as f:
        sets = f.read().split('\n')
    f.close()

    with open('{}/wiki_quality.txt'.format(args.entity_dir), 'r') as f:
        raw_list = f.read()
    f.close()

    entityset = set(raw_list.split('\n'))

    sets = [line for line in sets if line != '']

    num_query = args.num_query
    query_length = args.query_length
    eval_metric = {}

    query_set = []
    for entry in sets:
        query_set.append(json.loads(entry))

    #for item in query_set:
    item = query_set[0]

    score = 0
    recall = 0
    norm_score = 0
    query = ['belleville', 'kitchener', 'kenora']
    target = 'cities in ontario'
    # if np.count_nonzero(list(item['prob'].values())) < query_length * 2:
    #     continue

    print('prcessing query: ', query)
    labels = main_thrd(query, args.num_process, args.input_dir, target)
    top5 = [lab[0] for lab in labels[:5]]
    best_phrase = labels[0][0]
    best_sim = labels[0][1]['eval']
    length_labels = len(labels)
    recall_rank = int(np.argmax([lab[1]['eval'] for lab in labels]))
    recall_phrase = labels[recall_rank][0]
    recall_sim = labels[recall_rank][1]['eval']
    norm_best_sim = best_sim / recall_sim if recall_sim != 0 else 0
    recall += recall_sim
    score += best_sim
    norm_score += norm_best_sim
    meta = {'query':query, 'target': target, 'top1':(best_phrase, best_sim), 'top5': top5, 'recall':(recall_phrase, recall_rank+1, recall_sim), 'norm_top1': norm_best_sim}
    print(meta)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
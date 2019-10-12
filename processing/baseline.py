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
from itertools import product, combinations
from collections import Counter
from collections import defaultdict
from scipy.stats.mstats import gmean, hmean
from scipy.stats import skew, kurtosis
stop = set(stopwords.words('english'))
import nltk
import copy
from nltk.stem import WordNetLemmatizer
LEMMA = WordNetLemmatizer()

def sent_search(params):
    (task_list, query_iid, related_sent, input_dir) = params

    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    freq = dict()

    for ent in query_iid.keys():
        freq.update({ent:{'total':0}})

    context = dict((ent,[]) for ent in query_iid.keys())
    iid_set = set(related_sent.keys())

    subcorpus = []
    for fname in task_list:
       
        with open('{}/{}'.format(input_dir,fname), 'r') as f:
            for line in tqdm(f, desc='{}'.format(fname), mininterval=10):
                doc = json.loads(line)
                if doc['iid'] in iid_set:
                    subcorpus.append(doc)
        f.close()

    for item_dict in tqdm(subcorpus, desc='enrich-{}'.format(len(subcorpus)), mininterval=10):
        
        if item_dict['pid'] == 0 and item_dict['sid'] == 0: 
            doc = nlp(item_dict['text'])
            unigram = [token.lemma_ for token in textacy.extract.ngrams(doc,n=1, filter_nums=True, filter_punct=True, filter_stops=True, include_pos=["NOUN"])]
            item_dict['unigram'] = unigram
            tokens = [token.lemma_ for token in doc]
            item_dict['tokens'] = [token.lemma_ for token in doc if not token.is_punct]
            pos = [token.pos_ for token in doc]
        
            for ent in related_sent[item_dict['iid']]:

                if ent in item_dict['nsubj']:
                    context[ent].append(item_dict)

                    freq[ent]['total'] += 1
                    if item_dict['did'] in freq[ent]:
                        freq[ent][item_dict['did']] += 1
                    else:
                        freq[ent].update({item_dict['did']:1})
    
    return {'context':context, 'freq':freq}

def phrase_eval(params):
    list_phrases, unigram_set, target_token, idf, pid = params
    
    idf_list = [*idf]
    idf_set = set(idf_list)

    tokenizer = MWETokenizer(separator=' ')
    for e in unigram_set:
        tokenizer.add_mwe(nltk.word_tokenize(e))

    phrases_score = []
    for phrase in list_phrases:
        score = 0
        tokens = nltk.word_tokenize(phrase)
        if not set(tokens).issubset(idf_set):
            continue
        nonstop_tokens = [token for token in tokens if token not in stop]
        if len(nonstop_tokens) / len(tokens) <= 0.5:
            continue        
        
        vocab = set(target_token).union(set(tokens))
        vocab = list(vocab.intersection(idf_set))
        target_vec = [0] * len(vocab)
        phrase_vec = [0] * len(vocab)
        
        target_token_freq = dict(Counter(target_token))
        target_token_subset = list(set(vocab).intersection(set(target_token)))
        for token in target_token_subset:
            index = vocab.index(token)
            target_vec[index] = target_token_freq[token]/len(target_token) * idf[token]
        
        phrase_token_freq = dict(Counter(tokens))
        for token in tokens:
            index = vocab.index(token)
            phrase_vec[index] = phrase_token_freq[token]/len(tokens) * idf[token]
        
        tfidf_sim = 1 - spatial.distance.cosine(target_vec, phrase_vec)

        phrases_score.append([phrase, tfidf_sim])
    
    return phrases_score

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main_thrd(query_set, args, iindex):
    start_time = time.time()
    nlp = spacy.load('en_core_web_lg', disable=['ner'])
    
    unique_ent = set()

    for item in query_set:
        target = item['target']
        queries = item['queries']
        for query in queries:
            unique_ent = unique_ent.union(set(query))

    # ##### sentence search #####
    query_iid = {}
    related_sent = defaultdict(list)
    for ent in tqdm(unique_ent, desc='loading-entity', mininterval=10):
        mentions = set(iindex[ent])
        query_iid.update({ent:mentions})

    for k, v in tqdm(query_iid.items(), desc='related-sents', mininterval=10):
        for iid in v:
            related_sent[iid].append(k)

    input_files = os.listdir(args.input_dir)
    tasks = list(split(input_files, args.num_process))

    inputs = [(tasks[i], query_iid, related_sent, args.input_dir) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        search_results = p.map(sent_search, inputs)

    search_merge = search_results[0]['context']
    count_merge = search_results[0]['freq']

    for pid in range(1, len(search_results)):
        tmp_context = search_results[pid]['context']
        tmp_freq = search_results[pid]['freq']
        for ent in unique_ent:
            search_merge[ent] += tmp_context[ent]
            count_merge[ent]['total'] += tmp_freq[ent]['total']
            tmp_freq[ent].pop('total', None)
            count_merge[ent].update(tmp_freq[ent])

    for ent in unique_ent:
        for index in range(len(search_merge[ent])):
            search_merge[ent][index]['doc_score'] = count_merge[ent][search_merge[ent][index]['did']]/count_merge[ent]['total']

    print("--- search use %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()

    ### query processing ###
    num_query = args.num_query
    query_length = args.query_length
    eval_metric = {}
    bar = 1

    for item in query_set:
        top1_score = 0
        top5_score = 0
        top10_score = 0
        recall = 0
        norm_score = 0
        index = 0
        target = item['target']
        queries = item['queries']
        print('prcessing set: ', target)
        sys.stdout.flush()
        
        for query in queries:

            print('prcessing query: ', query)
            sys.stdout.flush()

            unigrams = []
            for ent in query:
                for sent in search_merge[ent]:
                    unigrams += sent['unigram']
            unigram_set = set(unigrams)

            print('(1/3) generate unigrams')
            sys.stdout.flush()

            N = 0
            cnt = Counter()
            for ent in query:
                N += len(search_merge[ent])
                for sent in search_merge[ent]:
                    cnt.update(sent['tokens'])
            cnt = dict(cnt)

            for ent in query:
                for word in nltk.word_tokenize(ent):
                    unigram_set.discard(word)
                    unigram_set.discard(LEMMA.lemmatize(word))

            idf = {}
            for key in cnt.keys():
                idf.update({key:np.log((N / cnt[key]))})

            print('(2/3) compute idf')
            sys.stdout.flush()

            context = ''
            for ent in query:
                context += ' '.join([sent['text'] for sent in search_merge[ent]])
            
            doc = nlp(context)
            tokens = [token.lemma_ for token in doc]
            pos = [token.pos_ for token in doc]
            phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos, minlen=1, maxlen=4)
            list_phrases = list(phrases['counts'])[:15]

            idf_list = [*idf]
            target_doc = nlp(target)
            target_token = [token.lemma_ for token in target_doc if not token.is_punct]

            params = (list_phrases, unigram_set, target_token, idf, 0)
            phrases_sorted = phrase_eval(params)

            print(phrases_sorted)
            print('(3/3) evaluate phrases')
            sys.stdout.flush()

            if phrases_sorted == []:
                continue

            top10 = [lab[0] for lab in phrases_sorted[:10]]
            best_phrase = phrases_sorted[0][0]
            best_sim = phrases_sorted[0][1]
            top5_sim = max([lab[1] for lab in phrases_sorted[:5]])
            top10_sim = max([lab[1] for lab in phrases_sorted[:10]])
            recall_rank = int(np.argmax([lab[1] for lab in phrases_sorted]))
            recall_phrase = phrases_sorted[recall_rank][0]
            recall_sim = phrases_sorted[recall_rank][1]
            norm_best_sim = best_sim / recall_sim if recall_sim != 0 else 0
            recall += recall_sim
            top1_score += best_sim
            top5_score += top5_sim
            top10_score += top10_sim
            norm_score += norm_best_sim
            meta = {'query':query, 'target': target, 'top10': top10, 'sim@1':best_sim, 'sim@5': top5_sim, 'sim@10': top10_sim, 'sim@full':(recall_phrase, recall_rank+1, recall_sim), 'norm_sim@1': norm_best_sim}
            print(meta)
            sys.stdout.flush()
            with open('{}/log-{}-{}.txt'.format(args.output_dir, query_length, args.sampling_method), 'a+') as f:
                f.write(json.dumps(meta) + '\n')
            f.close()
        
        top1_score /= num_query
        top5_score /= num_query
        top10_score /= num_query
        recall /= num_query
        norm_score /= num_query
        eval_metric.update({target:{'sim@1': top1_score, 'sim@5': top5_score, 'sim@10': top10_score, 'sim@full': recall, 'norm_sim@1': norm_score}})
        with open('{}/tfidf-sim-{}-{}.txt'.format(args.output_dir, query_length, args.sampling_method), 'a+') as f:
            f.write(json.dumps(eval_metric) + '\n')
        f.close()
        
        print('---- progess in {}/{} ----'.format(bar, len(query_set)))
        bar += 1
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="heuristic approach")
    parser.add_argument('--input_dir', type=str, default='', help='corpus directory')
    parser.add_argument('--query_dir', type=str, default='', help='search query')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    parser.add_argument('--num_query', type=int, default=5, help='number of query per set')
    parser.add_argument('--query_length', type=int, default=3, help='query length')
    parser.add_argument('--entity_dir', type=str, default='', help='entity files directory')
    parser.add_argument('--inverted_dir', type=str, default='', help='inverted index directory')
    parser.add_argument('--sampling_method', type=str, default='random', help='query sampling method')
    parser.add_argument('--output_dir', type=str, default='', help='output dict')
    
    args = parser.parse_args()

    with open('{}'.format(args.query_dir), 'r') as f:
        sets = f.read().split('\n')
    f.close()

    sets = [line for line in sets if line != '']

    query_set = []
    for entry in sets:
        query_set.append(json.loads(entry))

    with open('{}/wiki_quality.txt'.format(args.entity_dir), 'r') as f:
        raw_list = f.read()
    f.close()

    entityset = set(raw_list.split('\n'))

    with open('{}'.format(args.inverted_dir), "r") as f:
        raw = f.read()
    f.close()
    iindex = json.loads(raw)

    main_thrd(query_set, args, iindex)

if __name__ == '__main__':
    main()
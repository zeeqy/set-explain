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
import time
from scipy.stats import skew
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.stem.snowball import SnowballStemmer
stop = set(stopwords.words('english'))
stemmer = SnowballStemmer(language='english')

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

            entity_text = set([em for em in item_dict['entityMentioned']])

            for ent in query:
                if ent not in entity_text:
                    continue
                else:
                    doc = nlp(item_dict['text'])
                    if len(doc) >= 30:
                        continue
                    unigram = [token.text for token in textacy.extract.ngrams(doc,n=1,filter_nums=True, filter_punct=True, filter_stops=True)]
                    item_dict['unigram'] = unigram
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

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main_thrd(query, num_process, input_dir):
    start_time = time.time()
    nlp = spacy.load('en_core_web_lg', disable=['ner']) 
    nlp.max_length = 50000000

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

    # unigram_set = unigrams[0]
    # for item in unigrams:
    #     unigram_set = unigram_set.union(item)

    for ent in query:
        unigram_set.discard(ent)

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
                    #if sent['did'] not in did:
                        #score_dist[ug][ent] += sent['doc_score']
                    did.add(sent['did'])

    agg_score = {}
    for ug in score_dist.keys():
        tmp_res = [item[1] for item in score_dist[ug].items()]
        agg_score.update({ug: np.mean(tmp_res) - np.std(tmp_res)})


    score_sorted = sorted(agg_score.items(), key=lambda x: x[1], reverse=True)

    print("--- unigram score %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()
    
    # context = ''
    # for ent in query:
    #     context += ' '.join([item['text'] for item in search_merge[ent]])

    # doc = nlp(context)
    # tokens = [token.text for token in doc]
    # pos = [token.pos_ for token in doc]
    # mined_phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos, minlen=2, maxlen=8)
    # mined_phrases = mined_phrases['counts']
    
    start_time = time.time()
    
    mined_phrases = []
    for ent in query:
        for sent in search_merge[ent]:
            mined_phrases += sent['phrases']

    print("--- phrase mining %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()

    start_time = time.time()
    tokenizer = MWETokenizer(separator=' ')

    for e in unigram_set:
        tokenizer.add_mwe(nltk.word_tokenize(e))
    
    list_phrases = set(mined_phrases)
    phrases_score = {}
    for phrase in tqdm(list_phrases, desc='phrase-eval'):
        score = 0
        tokens = nltk.word_tokenize(phrase)
        nonstop_tokens = [token for token in tokens if token not in stop]
        if len(nonstop_tokens) / len(tokens) <= 0.5:
            continue
        raw_tokenized = tokenizer.tokenize(tokens)
        tokenized_set = set(raw_tokenized)
        for token in tokenized_set.intersection(unigram_set):
            score += agg_score[token]
        phrases_score.update({phrase:score/len(nonstop_tokens)})

    phrases_sorted = sorted(phrases_score.items(), key=lambda x: x[1], reverse=True)

    print("--- phrase eval use %s seconds ---" % (time.time() - start_time))

    return [phrase[0] for phrase in phrases_sorted[:5]]

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

    with open('{}/valid_set.txt'.format(args.query_dir), 'r') as f:
        sets = f.read().split('\n')
    f.close()

    with open('{}/wiki_quality.txt'.format(args.entity_dir), 'r') as f:
        raw_list = f.read()
    f.close()

    entityset = set(raw_list.split('\n'))

    sets = [line for line in sets if line != '']

    num_query = args.num_query
    query_length = args.query_length
    bleu_eval = {}
    smoothie = SmoothingFunction().method3 # NIST smoothing

    query_set = []
    for entry in sets:
        query_set.append(json.loads(entry))

    for item in query_set:
        score = 0
        seeds = [w.lower().replace('-', ' ').replace('_', ' ') for w in item['entities']]
        target = item['title'].lower().split(',')[0]
        target_token = [[stemmer.stem(token.text) for token in nlp(target)]]
        index = 0
        retry = 0
        while index < num_query:
            if retry > 1000:
                break
            query = list(np.random.choice(seeds, query_length))
            if len(set(query).intersection(entityset)) != len(query):
                retry += 1
                continue
            labels = main_thrd(query, args.num_process, args.input_dir)
            candidate = [stemmer.stem(token.text) for token in nlp(labels[0])]
            # for lab in labels:
            #     doc = nlp(lab)
            #     candidate.append([token.text for token in doc])
            bleu = sentence_bleu(target_token, candidate, smoothing_function=smoothie)
            score += bleu
            index += 1
            print(query, target_token, candidate, bleu)
        if retry > 1000:
            continue
        score /= num_query
        bleu_eval.update({target:score})

        with open('bleu-{}.txt'.format(query_length), 'a+') as f:
            f.write(json.dumps(bleu_eval) + '\n')
        f.close()

if __name__ == '__main__':
    main()
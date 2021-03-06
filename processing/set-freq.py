import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import copy
import time
from scipy.stats import skew, kurtosis

def sent_search(params):
    (task_list, query_set, input_dir) = params

    freq = dict()

    query_set_prob = copy.deepcopy(query_set)
    for i in range(len(query_set_prob)):
        query_set_prob[i].update({'prob':{}})
        for ent in query_set_prob[i]['entities']:
            query_set_prob[i]['prob'].update({ent:0})
            freq.update({ent:{'total':0}})

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
            for i in range(len(query_set_prob)):
                for ent in query_set_prob[i]['entities']:
                    if ent not in entity_text:
                        continue
                    else:
                        query_set_prob[i]['prob'][ent] += 1

                        freq[ent]['total'] += 1
                        if item_dict['did'] in freq[ent]:
                            freq[ent][item_dict['did']] += 1
                        else:
                            freq[ent].update({item_dict['did']:1})

    return {'set-prob': query_set_prob, 'freq':freq}

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main_thrd(query_set, num_process, input_dir):
    start_time = time.time()

    #regex = re.compile('[@_!#$%^&*()<>?/\|}{~:-,.]')

    query_set_prob = []
    for item in query_set:
        clean_set = []
        for ent in set(item['entities']):
            if all(x.isalpha() or x.isspace() for x in ent): 
                clean_set.append(ent.lower())
        if len(clean_set) > 7 and all(x.isalpha() or x.isspace() for x in item['title']):
            item['title'] = item['title'].lower()
            item['entities'] = clean_set
            query_set_prob.append(item)

    ##### sentence search #####
    input_files = os.listdir(input_dir)
    tasks = list(split(input_files, num_process))
    
    inputs = [(tasks[i], query_set_prob, input_dir) for i in range(num_process)]

    with Pool(num_process) as p:
        search_results = p.map(sent_search, inputs)
    
    search_merge = search_results[0]['set-prob']
    count_merge = search_results[0]['freq']
    
    for pid in range(1, len(search_results)):
        for i in range(len(query_set_prob)):
            for ent in search_merge[i]['prob'].keys():
                search_merge[i]['prob'][ent] += search_results[pid]['set-prob'][i]['prob'][ent]

    for pid in range(1, len(search_results)):
        tmp_freq = search_results[pid]['freq']
        for ent in count_merge.keys():
            count_merge[ent]['total'] += tmp_freq[ent]['total']
            tmp_freq[ent].pop('total', None)
            count_merge[ent].update(tmp_freq[ent])
    
    skew_dict = {}
    for ent in count_merge.keys():
        skew_list = []
        did_list = [did for did in  count_merge[ent] if did != 'total']
        for did in did_list:
            weight = count_merge[ent][did]/count_merge[ent]['total']
            skew_list += [weight]*count_merge[ent][did]
        if len(skew_list) > 1:
            skew_list.sort(reverse=True)
            skew_dict.update({ent:skew(skew_list)})
        else:
            skew_dict.update({ent:0})

    results = []
    for item in search_merge:
        try:
            item['freq'] = item['prob']
            item['prob'] = {k: v / total for total in (sum(item['prob'].values()),) for k, v in item['prob'].items()}
            item['skew'] = {k: skew_dict[k] for k, v in item['prob'].items()}
            results.append(item)
        except:
            print(item)

    print("--- search use %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()

    return results

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

    results = main_thrd(query_set, args.num_process, args.input_dir)

    with open('{}/set_prob_large.txt'.format(args.query_dir), 'w+') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
    f.close()

if __name__ == '__main__':
    main()
import json, sys, os, re
import numpy as np
import argparse
import random
import bisect

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]


def main():
    parser = argparse.ArgumentParser(description="query sampler")
    parser.add_argument('--query_dir', type=str, default='', help='search query')
    parser.add_argument('--sampling_method', type=str, default='random', help='query sampling method')
    parser.add_argument('--query_length', type=int, default=3, help='query length')
    parser.add_argument('--num_query', type=int, default=5, help='number of query per set')
    parser.add_argument('--output_dir', type=str, default='', help='output dir')

    args = parser.parse_args()

    with open('{}'.format(args.query_dir), 'r') as f:
        sets = f.read().split('\n')
    f.close()

    sets = [line for line in sets if line != '']

    query_set = []
    for entry in sets:
        query_set.append(json.loads(entry))
    
    num_query = args.num_query
    query_length = args.query_length
    sampling_method = args.sampling_method

    query_data = []
    for item in query_set:
        if np.count_nonzero(list(item['prob'].values())) < 7:
            continue
        if sampling_method == 'freq':
            queries = [np.random.choice(list(item['prob'].keys()), query_length, replace=False, p=list(item['prob'].values())).tolist() for i in range(num_query)]
        if sampling_method == 'random':
            valid_ent = [ent[0] for ent in item['prob'].items() if ent[1] > 0]
            queries = [np.random.choice(valid_ent, query_length, replace=False).tolist() for i in range(num_query)]
        if sampling_method == 'skew':
            valid_ent = [ent[0] for ent in item['prob'].items() if ent[1] > 0]
            queries = []
            for j in range(num_query):
                skew_dist = item['skew']
                query = []
                for i in range(query_length)
                    ent = choice(list(skew_dist.keys()), skew_dist.values())
                    skew_dist.pop(ent,None)
                    query.append(ent)
                queries.append(query)

        query_data.append({'target': item['title'].lower().split(',')[0], 'queries': queries})
    
    with open('{}/query-{}-{}.txt'.format(args.output_dir, query_length, sampling_method), 'a+') as f:
        for q in query_data:
            f.write(json.dumps(q) + '\n')
    f.close()

if __name__ == '__main__':
    main()
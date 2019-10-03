import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import string
import phrasemachine
import textacy
import spacy
from nltk.tokenize import MWETokenizer
import nltk

"""
create inverted index

"""

def merge_task(params):
    task_list, args = params

    with open('{}/wiki_quality.txt'.format(args.entity_dir), 'r') as f:
        raw_list = f.read()
    f.close()

    entityset = set(raw_list.split('\n'))

    print("successfully read entity file and initialized tokenizer")
    sys.stdout.flush()

    context = dict.fromkeys(entityset, [])

    for fname in task_list:
        outputname = 'INVERTED_INDEX_{}'.format(fname.split('_')[-1])

        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in tqdm(doc, desc='{}'.format(fname), mininterval=30):
            item_dict = json.loads(item)
            for ent in item_dict['entityMentioned']:
                context[ent].append(item_dict['iid'])

    return context

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
    parser = argparse.ArgumentParser(description="inverted index")
    parser.add_argument('--input_dir', type=str, default='', help='json document directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    parser.add_argument('--entity_dir', type=str, default='', help='entity files directory')
    
    args = parser.parse_args()

    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    inputs = [(tasks[i], args) for i in range(args.num_process)]

    with Pool(args.num_process) as p:
        merge_results = p.map(merge_task, inputs)

    with open('{}/wiki_quality.txt'.format(args.entity_dir), 'r') as f:
        raw_list = f.read()
    f.close()

    entityset = set(raw_list.split('\n'))

    inverted_index = dict.fromkeys(entityset, [])

    for key in tqdm(inverted_index.keys(), desc='merge', mininterval=10):
        for res in merge_results:
            inverted_index[key] += res[key]

    print(len(inverted_index['boston']))
    sys.stdout.flush()

    with open('{}/inverted_index.txt'.format(args.output_dir), "w+") as f:
        json.dump(inverted_index, f)
    f.close()

if __name__ == '__main__':
    main()
    
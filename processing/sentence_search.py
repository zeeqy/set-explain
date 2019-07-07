import json, sys, os, re
import argparse
import bisect
import nltk
import multiprocessing as mp
import spacy
from tqdm import tqdm

"""
search sentence based on keywords

"""

def merge_task(task_list, args, outputs):
    nlp = spacy.load("en_core_web_lg", disable=['ner'])
    kwlt = args.keywords.split(',')
    kwset = set(kwlt)
    context = []
    for fname in task_list:

        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in tqdm(doc, desc='{}'.format(fname), mininterval=30):
            try:
                item_dict = json.loads(item)
            except:
                print(fname, item)
                sys.stdout.flush()
                continue
            entity_text = set([em for em in item_dict['entityMentioned']])
            if entity_text.intersection(kwset) == kwset:
                doc = nlp(item_dict['text'])
                nsubj = [chunk.text for chunk in doc.noun_chunks if chunk.root.dep_ == 'nsubj']
                if kwlt[0] in nsubj:
                    context.append(item_dict)

    outputs.put(context)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
    parser = argparse.ArgumentParser(description="search sentence based on keywords")
    parser.add_argument('--input_dir', type=str, default='', help='autophrase parsed directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    parser.add_argument('--output_prefix', type=str, default='', help='output filename')
    parser.add_argument('--keywords', type=str, default='', help='search keywords')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    
    args = parser.parse_args()

    outputs = mp.Queue()
    search_results = []

    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    processes = [mp.Process(target=merge_task, args=(tasks[i], args, outputs)) for i in range(args.num_process)]

    for p in processes:
        p.start()
        search_results += outputs.get()
        

    for p in processes:
        p.join()
        
    with open('{}/{}'.format(args.output_dir, args.output_prefix), "w+") as f:
        f.write('\n'.join([json.dumps(res) for res in search_results]))
    f.close()

if __name__ == '__main__':
    main()
    
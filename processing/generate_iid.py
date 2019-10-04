import json, sys, os, re
import argparse
import bisect
import multiprocessing as mp
from tqdm import tqdm
import string
import phrasemachine
import textacy
import spacy
from nltk.tokenize import MWETokenizer
import nltk

pronoun = set(["he", "her", "hers", "herself", "him", "himself", "his", "i", 
"it", "its", "me", "mine", "my", "myself", "our", "ours", "ourselves", 
"she", "thee", "their", "them", "themselves", "they", "thou", 
"thy", "thyself", "us", "we", "ye", "you", "your", "yours", "yourself",
"we", "anybody", "anyone", "anything", "each", "either", "everybody", 
"everyone", "everything", "neither", "nobody", "no one", "nothing", "one", "somebody", 
"someone", "something", "all", "any", "most", "none", "some", "both", "few", "many", "several",
 "what", "who", "which", "whom", "whose", "that", "whichever", "whoever", "whomever", 
 "this", "these", "that", "those"])

"""
create inverted index

"""

def merge_task(task_list, args):

    for fname in task_list:
        outputname = 'INVERTED_INDEX_{}'.format(fname.split('_')[-1])
        context = []#dict.fromkeys(entityset, [])

        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in tqdm(doc, desc='{}'.format(fname), mininterval=30):
            item_dict = json.loads(item)
            if set(item_dict['nsubj']).intersection(pronoun) != set():
                continue
            item_dict['iid'] = '{}{}{}'.format(item_dict['did'],item_dict['pid'],item_dict['sid'])
            context.append(json.dumps(item_dict))

        with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
            f.write('\n'.join(context))
        f.close()

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

    processes = [mp.Process(target=merge_task, args=(tasks[i], args)) for i in range(args.num_process)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
    
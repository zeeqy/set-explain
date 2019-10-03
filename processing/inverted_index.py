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

def main():
    parser = argparse.ArgumentParser(description="inverted index")
    parser.add_argument('--input_dir', type=str, default='', help='json document directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    parser.add_argument('--entity_dir', type=str, default='', help='entity files directory')
    
    args = parser.parse_args()

    input_dir = os.listdir(args.input_dir)

    with open('{}/wiki_quality.txt'.format(args.entity_dir), 'r') as f:
        raw_list = f.read()
    f.close()

    entityset = set(raw_list.split('\n'))

    print("successfully read entity file and initialized tokenizer")
    sys.stdout.flush()

    context = dict.fromkeys(entityset, [])

    for fname in tqdm(input_dir, desc='file', mininterval=10):

        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in doc:
            item_dict = json.loads(item)
            for ent in item_dict['entityMentioned']:
                context[ent].append(item_dict['iid'])

    print(len(context['boston']))

    # with open('{}/inverted_index.txt'.format(args.output_dir), "w+") as f:
    #     for key, values in tqdm(context.items(), desc='dump', mininterval=10):
    #         f.write(json.dumps({key:values}) + '\n')
    # f.close()

if __name__ == '__main__':
    main()
    
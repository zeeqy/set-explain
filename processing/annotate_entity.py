import json, sys, os, re
import argparse
import bisect
import threading
from nltk.tokenize import MWETokenizer

"""
find entities mentioned in each sentence

"""

def merge_task(task_list, args):
    with open('{}/entitylist.txt'.format(args.entity_dir), 'r') as f:
        raw_list = f.readlines()
    f.close()

    entityset = set(raw_list)

    with open('{}/entity2id.txt'.format(args.entity_dir), 'r') as f:
        raw_entity2id = f.read()
    f.close()

    entity2id = json.loads(raw_entity2id)

    with open('{}/id2entity.txt'.format(args.entity_dir), 'r') as f:
        raw_id2entity = f.read()
    f.close()

    id2entity = json.loads(raw_id2entity)

    tokenizer = MWETokenizer(separator=' ')

    for e in entity_list:
        tokenizer.add_mwe(e.split())

    print("successfully read entity file and initialized tokenizer")
    sys.stdout.flush()

    for fname in task_list:
        outputname = 'SENTENCE_ENTITY_{}'.format(fname.split('_')[-1])
        context = {}

        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in doc:
            item_dict = json.loads(item)
            raw_tokenized = tokenizer.tokenize(item_dict['text'].split())
            tokenized_set = set(raw_tokenized)
            mentioned_entity = tokenized_set.intersection(entityset)
            mentioned2id = [entity2id[e] for e in mentioned_entity]
            label = "{}-{}-{}".format(item_dict['did'],item_dict['pid'],item_dict['sid'])
            item_dict.update({'mentioned':mentioned2id})
            context.update({label:item_dict})

        with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
            f.write(json.dumps(context))
        f.close()

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
    parser = argparse.ArgumentParser(description="Keep json format and clean text ")
    parser.add_argument('--input_dir', type=str, default='', help='json document directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    parser.add_argument('--entity_dir', type=str, default='', help='entity files directory')
    
    args = parser.parse_args()

    input_dir = os.listdir(args.input_dir)
    tasks = list(split(input_dir, args.num_process))

    threads = []
    for i in range(args.num_process):
        t = threading.Thread(target=merge_task, args=(tasks[i], args, ))
        threads.append(t)
        t.start()

if __name__ == '__main__':
    main()
    
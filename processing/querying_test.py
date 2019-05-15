import json, sys, os
import argparse

"""
find entities mentioned in each sentence

"""
def main():
    parser = argparse.ArgumentParser(description="Keep json format and clean text ")
    parser.add_argument('--sentence_dir', type=str, default='', help='sentence document directory')
    parser.add_argument('--entity_dir', type=str, default='', help='entity document directory')
    parser.add_argument('--inverted_dir', type=str, default='', help='entity document directory')
    parser.add_argument('--input_entity', type=str, default='', help='querying entity')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    
    args = parser.parse_args()

    with open('{}/INVERTED_INDEX.json'.format(args.inverted_dir), 'r') as f:
        raw_index = f.read()
    f.close()
    inverted_index = json.loads(raw_index)

    with open('{}/entity2id.txt'.format(args.entity_dir), 'r') as f:
        raw_entity2id = f.read()
    f.close()

    entity2id = json.loads(raw_entity2id)

    eid = '{}'.format(entity2id[args.input_entity])

    print(inverted_index[eid])

if __name__ == '__main__':
    main()
    
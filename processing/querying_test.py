import json, sys, os
import argparse
from querying_tools import matching_tools

"""
find entities mentioned in each sentence

"""
def main():
    parser = argparse.ArgumentParser(description="Keep json format and clean text ")
    parser.add_argument('--sentence_dir', type=str, default='', help='sentence document directory')
    parser.add_argument('--entity_dir', type=str, default='', help='entity document directory')
    parser.add_argument('--inverted_dir', type=str, default='', help='inverted document directory')
    parser.add_argument('--query_entity', type=str, default='', help='querying entity')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    #parser.add_argument('--num_process', type=int, default=2, help='number of parallel')
    
    args = parser.parse_args()

    outputname = 'query_test_output.txt'

    mtools = matching_tools(args.entity_dir, args.inverted_dir, args.sentence_dir)

    print("initialization finished")
    sys.stdout.flush()

    mentioned_keys = mtools.entityMentioned(args.query_entity)

    print("search for entity mention finished")
    sys.stdout.flush()

    content = mtools.key2Text(mentioned_keys)

    with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
        f.write('\n'.join(content))
    f.close()

if __name__ == '__main__':
    main()
    
import json, sys, os
import argparse

"""
create inverted index

"""
def main():
    parser = argparse.ArgumentParser(description="Keep json format and clean text ")
    parser.add_argument('--input_dir', type=str, default='', help='json document directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    
    args = parser.parse_args()

    task_list = os.listdir(args.input_dir)

    num_file = len(task_list)

    entity_dict = {}

    outputname = 'INVERTED_INDEX.txt'

    count = 0

    for fname in task_list:

        print("start processing {}".format(fname))
        sys.stdout.flush()
        
        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in doc:
            item_dict = json.loads(item)
            seid = list(item_dict.keys())[0]
            mentioned2id = item_dict[seid]['mentioned']
            for eid in mentioned2id:
                if eid in entity_dict.keys():
                    entity_dict[eid].append(seid)
                else:
                    entity_dict.update({eid:[seid]})
        
        count += 1
        print("finished processing {}, {}/{}".format(fname, count, num_file))
        sys.stdout.flush()

    context = []

    for eid in entity_dict.keys():
        context.append(json.dumps({eid:entity_dict[eid]}))
    
    with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
        f.write('\n'.join(context))
    f.close()

if __name__ == '__main__':
    main()
    
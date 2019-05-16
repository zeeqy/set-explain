import json, sys, os
import argparse

"""
convert sentence json to csv

"""
def main():
    parser = argparse.ArgumentParser(description="Convert sentence json to csv")
    parser.add_argument('--input_dir', type=str, default='', help='json document directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')
    
    args = parser.parse_args()

    task_list = os.listdir(args.input_dir)

    num_file = len(task_list)

    count = 0

    for fname in task_list:

        context = []

        outputname = 'SENTENCE_ENTITY_{}.csv'.format(fname.split('_')[-1])

        print("start processing {}".format(fname))
        sys.stdout.flush()
        
        with open('{}/{}'.format(args.input_dir,fname), 'r') as f:
            doc = f.readlines()
        f.close()

        for item in doc:
            item_dict = json.loads(item)
            mid = list(item_dict.keys())[0]
            if len(item_dict[mid]['mentioned']) != 0:
                mentioned = ','.join([str(eid) for eid in item_dict[mid]['mentioned']])
                test = item_dict[mid]['text'].replace('|','')
                ln = """{}|{}|{}|{}|{}|{}|{}""".format(mid, item_dict[mid]['title'], item_dict[mid]['did'], item_dict[mid]['pid'], item_dict[mid]['sid'], mentioned, text)
                context.append(ln)

        count += 1
        print("finished processing {}, {}/{}".format(fname, count, num_file))
        sys.stdout.flush()
    
        with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
            f.write('\n'.join(context))
        f.close()

if __name__ == '__main__':
    main()
    
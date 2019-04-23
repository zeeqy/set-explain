import sys, os, time
import argparse


parser = argparse.ArgumentParser(description="Page views collect")
parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
parser.add_argument('--output_dir', type=str, default='', help='output directory')
args = parser.parse_args()

file_list = os.listdir(args.input_dir)

for fname in file_list:
	in_dir = '{}/{}'.format(args.input_dir,fname)
	os.system("python3 /root/set-explain/processing/pages_views.py --input_dir={} --output_dir={}".format(in_dir,args.output_dir))
	print('finish running {}'.format(fname))
	time.sleep(0.1)  
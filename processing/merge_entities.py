import json, sys, os, re
import argparse
import numpy as np
import pandas as pd

"""
1. Merge entities from AutoPhrase and wiki document title

2. Generate a high quality entity list  

"""
def parse(title):
	return re.sub(r'\([^)]*\)', '', title).strip()

def main():
	parser = argparse.ArgumentParser(description="Merge and clean entities")
	parser.add_argument('--input_dir', type=str, default='', help='dump file directory')
	parser.add_argument('--output_dir', type=str, default='', help='output directory')
	parser.add_argument('--cutoff', type=float, default=0.25, help='AutoPhrase score cutoff')
	
	args = parser.parse_args()

	title_fname = 'concat_title.txt'
	phrase_fname = 'AutoPhrase.txt'
	outputname = 'wiki_entities.txt'

	with open('{}/{}'.format(args.input_dir,title_fname), 'r') as f:
		title_raw = f.readlines()
	f.close()
	title_clean = []
	for t in title_raw:
		title_clean.append(parse(t))
	title_set = set(title_clean)

	phrase_raw = pd.read_csv('{}/{}'.format(args.input_dir, phrase_fname),header=None, delimiter='\t')
	phrase_raw.columns = ['score','entity']

	score_cutoff = phrase_raw[phrase_raw.score >= args.cutoff].entity.values.tolist()

	phrase_set = set(phrase_raw.entity.values.tolist())
	phrase_set.remove(np.nan)
	title_entity = list(phrase_set.intersection(title_set))

	overall_entity = list(set(score_cutoff + title_entity))

	with open('{}/{}'.format(args.output_dir, outputname), "w+") as f:
			f.write('\n'.join([str(t) for t in overall_entity]))
	f.close()

if __name__ == '__main__':
	main()
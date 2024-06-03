#!/usr/bin/python

#USAGE: python read_blast_output.py --sequences SEQUENCES_FASTA --assembly ASSEMBLY_FASTA \
#                                  	   --mapping BLAST_OUTPUT_TSV --threshold PID_THR \
#									   --outdir OUT_DIR --outfile OUT_FILENAME	

from __future__ import division
import os
import argparse
import pandas as pd
from Bio import SeqIO

#Read BLAST output file into table
def read_blast_output(file):
	col_names = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"]  # outfmt 6
	return pd.read_csv(file, sep = '\t', names = col_names, dtype = str)

# compute number of positions (integers) covered by a list of potentially overlapping intervals
def num_covered_positions(intervals):
	intervals.sort(key = lambda x: x[0])  # intervals is now sorted by start position
	num_pos_covered = 0
	last_pos_covered = 0  # last (right-most) position of contig covered so far
	for start, end in intervals:
		if end <= last_pos_covered:
			pass  # contained in previous interval -> no new position covered
		else:
			num_pos_covered += end - max(last_pos_covered + 1, start) + 1
			last_pos_covered = end
	return num_pos_covered

#Analyse the matches between the genes / proteins found in plasmids and the reference contigs
def analyse_from_contigs(hits, plsness_seqs, contig_seqs, threshold, out):
	'''
	hits: dataframe of blast output
	plsness_seqs: dictionary of genes / proteins with sequence lengths as values 
	contig_seqs: dictionary of contigs with contig lengths as values
	threshold: pident threshold
	out: path to output file
	'''
	covered_sections = dict([(pred, []) for pred in plsness_seqs])	
	covered_per_ref = dict()		#Covered proportion per reference plasmid

	ref_ids = hits.sseqid.unique()	#Set of reference ids in the blast output

	for ref in sorted(ref_ids):
		covered_per_ref[ref] = {}
		for seq in plsness_seqs:
			covered_per_ref[ref][seq] = []

			seq_ref_hits = hits.loc[hits.qseqid == seq].loc[hits.sseqid == ref]
			for index, row in seq_ref_hits.iterrows():
				qstart, qend = int(row[6]), int(row[7])
				pident = float(row[2])/100
				interval = (qstart, qend) if qstart <= qend else (qend, qstart)
				if pident >= threshold:
					covered_sections[seq].append(interval)
					covered_per_ref[ref][seq].append(interval)

	
	#Outputting to file with tsv format:
	#Ref_ID\y	Ctg_ID	Percent_mapping	Ref_len	Ctg_len 
	for ref in covered_per_ref:
		for seq in covered_per_ref[ref]:
			seq_len = len(str(plsness_seqs[seq]))

			percent_mapping = num_covered_positions(covered_per_ref[ref][seq])/seq_len
			ref_len = plasmid_seqs[ref]
			if percent_mapping >= 0.8:
				out.write(ref + '\t' + seq + '\t' + "{:.2f}".format(percent_mapping) + '\t' + str(ref_len) + '\t' + str(seq_len) + "\n")
	   
argparser = argparse.ArgumentParser()
argparser.add_argument('--sequences', help = 'sequences FASTA file')
argparser.add_argument('--assembly', help = 'assembly FASTA file')
argparser.add_argument('--mapping', help = 'blast output TSV file')
argparser.add_argument('--threshold', help = 'percent identity threshold')
argparser.add_argument('--outdir', help = 'output directory')
argparser.add_argument('--outfile', help = 'name of output TSV file')


args = argparser.parse_args()

SEQ_FASTA = args.sequences
CTG_FASTA = args.assembly
BLAST_OUT = args.mapping
OUT_DIR = args.outdir
OUT_FILENAME = args.outfile
PID_THR = float(args.threshold) 

mapping = read_blast_output(BLAST_OUT)

ctg_seqs = dict()
with open(CTG_FASTA, "r") as f:
	for record in SeqIO.parse(f, "fasta"):
		ctg_seqs[record.id] = record.seq

plsness_seqs = dict()
with open(SEQ_FASTA, "r") as f:
	for record in SeqIO.parse(f, "fasta"):
		plsness_seqs[record.id] = len(record.seq)			

os.system('mkdir -p '+OUT_DIR)
OUT_FILE = OUT_DIR + '/' + OUT_FILENAME
with open(OUT_FILE, "w") as out:
	analyse_from_contigs(mapping, plsness_seqs, ctg_seqs, PID_THR, out)
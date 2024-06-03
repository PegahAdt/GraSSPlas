import os
import argparse

from gfa_fasta_utils import *

import pandas as pd
import statistics

argparser = argparse.ArgumentParser()
argparser.add_argument('--assembly', help = 'assembly GFA file')
argparser.add_argument('--plsness', help = 'folder containing plasmidness files (hits between genes and contigs)')
argparser.add_argument('--len_threshold', help = 'length threshold for chromosome pseudolabels')
#argparser.add_argument('--gd_file', help = 'gene density file')
#argparser.add_argument('--gd_threshold', help = 'gene density threshold for plasmid pseudolabels by gd')
argparser.add_argument('--outdir', help = 'output directory')


args = argparser.parse_args()

CTG_GFA = args.assembly
PLSNESS_DIR = args.plsness
CHR_LEN_THR = int(args.len_threshold)
#GD_TSV = args.gd_file
#PLS_GD_THR = float(args.gd_threshold)
OUT_DIR = args.outdir

if os.path.exists(CTG_GFA):
    cov_key, len_key = "dp", "LN"
    ctg_dict = read_GFA_ctgs(in_file_path=CTG_GFA, attributes_list=[cov_key, len_key])
    ctg_df = pd.DataFrame.from_dict(ctg_dict).T
    mean_cov = ctg_df["dp"].mean()
    stdev_cov = ctg_df["dp"].std()

    GTYPES = ['replicon', 'origintransfer', 'mobility', 'matepair']
    plsness_set = set()
    for gtype in GTYPES:
        plsness_file = PLSNESS_DIR + '/' + gtype + '_mapping.tsv'
        if os.path.getsize(plsness_file) > 0:
            plsness_df = pd.read_csv(plsness_file, sep='\t', header=None)
            for index, row in plsness_df.iterrows():
                ctg_id = str(row[0])
                plsness_set.add(ctg_id)    

    '''
    gd_set = set()
    gd_df = pd.read_csv(GD_TSV, sep='\t', header=None) 
    for index, row in gd_df.iterrows():
        ctg_id, ctg_gd = str(row[0]), row[1]
        if ctg_gd >= 0.8 and ctg_dict[ctg_id][len_key] >= 500:
            gd_set.add(ctg_id)
    '''

    #gd_labels_dict = {}
    plsness_labels_dict = {}
    chr_set = set()
    for ctg_id in ctg_dict:
        #gd_labels_dict[ctg_id] = {'chromosome': 0, 'plasmid': 0}
        plsness_labels_dict[ctg_id] = {'chromosome': 0, 'plasmid': 0}
        if ctg_dict[ctg_id][len_key] >= 200000 and ctg_dict[ctg_id][cov_key] < (mean_cov + 2*stdev_cov):
            chr_set.add(ctg_id)
            #gd_labels_dict[ctg_id]['chromosome'] = 1
            plsness_labels_dict[ctg_id]['chromosome'] = 1
        #if ctg_id in gd_set:
        #    gd_labels_dict[ctg_id]['plasmid'] = 1
        if ctg_id in plsness_set:
            plsness_labels_dict[ctg_id]['plasmid'] = 1      
                
    #gd_labels_df = pd.DataFrame.from_dict(gd_labels_dict).T
    plsness_labels_df = pd.DataFrame.from_dict(plsness_labels_dict).T

    #gd_labels_df.index.names = ['contig']
    plsness_labels_df.index.names = ['contig']

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    #gd_labels_df.to_csv(OUT_DIR+'/pseudolabels_by_gd.tsv', sep='\t')
    plsness_labels_df.to_csv(OUT_DIR+'/pseudolabels.tsv', sep='\t')
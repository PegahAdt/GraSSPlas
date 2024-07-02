#!/bin/bash

# Specify the parent directory
PARENT_DIR="/project/ctb-chauvec/PLASMIDS/DATA/Ecoli_PlasEval_22May2024"

# Find all short_read.fasta.gz files and unzip them
find "$PARENT_DIR" -type f -name 'short_read.gfa.gz' -execdir gunzip -k {} \;
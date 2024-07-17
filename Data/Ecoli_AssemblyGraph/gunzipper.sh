#!/bin/bash

# Specify the parent directory
PARENT_DIR="/Users/mrsadeghian/Desktop/MrS/Research/GraSSPlas/Data/Ecoli_AssemblyGraph"

# Find all short_read.fasta.gz files and unzip them
find "$PARENT_DIR" -type f -name 'short_read.gfa.gz' -execdir gunzip -k {} \;
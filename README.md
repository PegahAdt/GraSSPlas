# Graph-Based Self-Supervised Learning for Repeat Detection in Metagenomic Assembly

This repository contains all necessary code and commands to detect the repeated sequences in a genomic dataset.

Here we propose a novel approach that leverages the assembly graph’s structure through graph neural networks (GNNs) within a self-supervised
learning framework to classify DNA sequences (unitigs) into repetitive and non-repetitive categories.
We frame this problem as a node classification task within the assembly graph.


## Installation

The code is based on Python 3.7 and should run on Unix-like operating systems (MacOS, Linux).

### Python libraries

Make sure you have the python packages listed in `requirements.txt` installed. You can install them using the following command:

```sh
$ pip install -r requirements.txt
```

### Packages:

In addition, ensure that you have installed these required packages:

- **Wgsim**: Follow the installation instructions provided in the [wgsim repository](https://github.com/lh3/wgsim).
- **ABySS**: Follow the installation instructions provided in the [Abyss repository](https://github.com/bcgsc/abyss).
- **Bowtie2**: Follow the installation instructions provided on the [bowtie2 website](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml).
- **{MUM}mer**: Follow the installation instructions provided in the [mummer repository](https://mummer4.github.io/).


## Running the Codes

1. **Folder Structure:**

You should have the following directory structure in the project folder:

```
├──Codes.py
├──Results
├── Data
│ ├── simulated_L400_C25
│ │ ├── (read pairs and reference genome)
│ ├── shakya_1
│ │ ├── (read pairs and reference genome)
│ ├── (other folders for different cases)
│ │ ├── (read pairs and reference genome)
```

You need to place your data files, including read pairs, in `.fq` format and reference genome in `.fasta` format in the respective folders inside the `Data` directory.

You need to have three main files provided, for example for shakya_1 dataset:

```
├── Data
│ ├── shakya_1
│ │ ├── outRead1.fq
│ │ ├── outRead2.fq
│ │ ├── ref_genome.fasta
```

2. **Running the Codes:**

Execute the `codes.sh` script to run the setup for each case. Make sure you are in the project root directory and run the following command:

```bash
bash shakya_code.sh
bash simulated.sh
```

This script will process the data files located in the `Data` directory, generate results for each setup specified in the script, and save them in the corresponding folder in the `Results` folder.





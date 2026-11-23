#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --account=def-chauvec
#SBATCH --output=logs/GraSSPlas_%a.out  # Output log file
#SBATCH --error=logs/GraSSPlas_%a.err   # Error log file
#SBATCH --mail-user=paa40@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-5  # Number of parallel jobs (adjust as needed)

# Set TOOLS_DIR to a specific folder in your home directory or a writable project directory
TOOLS_DIR="/home/paa40/projects/ctb-chauvec/paa40"
OUTPUT_DIR="/home/paa40/projects/ctb-chauvec/paa40/results_${SLURM_ARRAY_TASK_ID}"

# Load necessary modules
module load StdEnv/2020
module load python/3.8.10
module load cuda/11.0

# Debugging: Print environment and paths
echo "TOOLS_DIR: $TOOLS_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Ensure the TOOLS_DIR and OUTPUT_DIR exist and have the correct permissions
mkdir -p $TOOLS_DIR
mkdir -p $OUTPUT_DIR
chmod 700 $TOOLS_DIR
chmod 700 $OUTPUT_DIR

# List the contents of the tools and output directories
echo "Contents of TOOLS_DIR:"
ls -l $TOOLS_DIR
echo "Contents of OUTPUT_DIR before running the script:"
ls -l $OUTPUT_DIR

# Create and activate a virtual environment in the project space
python -m venv $TOOLS_DIR/venv
source $TOOLS_DIR/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install compatible versions of required packages from PyPI
pip install torch==1.13.1+computecanada torchvision==0.14.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+computecanada.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+computecanada.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+computecanada.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+computecanada.html
pip install torch-geometric

pip install pyfastg biopython numpy==1.21.0 scikit-learn==0.24.2 pandas networkx==2.6.3 matplotlib==3.7.5 matplotlib_venn
pip install upsetplot
pip install venn
pip install --upgrade matplotlib

# List installed packages
pip list

# Run your code
#INPUT_PATH="/home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/GrassPlas_Paths_New_Features.txt"
#INPUT_PATH="/home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/Paths/grassplas_paths.txt"
#INPUT_PATH="/home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/Paths/grassplas_filteredGFA_Pfam.txt"
INPUT_PATH="/home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/Paths/grassplas_filteredGFA_pseudo_060.txt"
#INPUT_PATH="/home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/Paths/grassplas_filteredGFA_paths.txt"
#INPUT_PATH="/home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/Paths/grassplas_filteredGFA_combined.txt"

echo "Running mainPlasDetection.py with INPUT_PATH: $INPUT_PATH"
python mainPlasDetection.py --name $INPUT_PATH --output_dir $OUTPUT_DIR

# List the contents of the output directory after running the script
echo "Contents of OUTPUT_DIR after running the script:"
ls -l $OUTPUT_DIR

# Deactivate the virtual environment
deactivate


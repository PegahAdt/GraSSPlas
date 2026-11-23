#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=40G
#SBATCH --account=def-chauvec
#SBATCH --output=GraSSPlas.out
#SBATCH --error=GraSSPlas.err
#SBATCH --mail-user=paa40@sfu.ca
#SBATCH --mail-type=ALL


# Set TOOLS_DIR to a specific folder in your home directory or a writable project directory
TOOLS_DIR="/home/paa40/projects/ctb-chauvec/paa40"
module load StdEnv/2020
module load python/3.8.10
module load cuda/11.0

# Ensure the TOOLS_DIR exists and has the correct permissions
mkdir -p $TOOLS_DIR
chmod 700 $TOOLS_DIR

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
pip install matplotlib upsetplot
pip install venn
pip install --upgrade matplotlib
# Install torch-geometric and its dependencies using specific URLs to ensure compatibility
#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric


# Run your code
python mainPlasDetection.py --name /home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/GrassPlas_Paths_New_Features.txt
#python mainPlasDetection.py --name /home/paa40/projects/ctb-chauvec/PLASMIDS/DATA/new_grassplas_paths.txt
#python test_mock.py --name /home/paa40/projects/ctb-chauvec/paa40/mock_path.txt
# Deactivate the virtual environment
deactivate


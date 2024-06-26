import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from venn import venn

# Function to process each file and return Venn data
def process_file(file_path):
    data = pd.read_csv(file_path, sep='\t')
    tools = ['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls']
    plasmid_classification = {}

    for tool in tools:
        plasmid_classification[tool] = set(data['contig'][data[tool] >= 0.5])

    venn_data = {
        'Plasforest': plasmid_classification['plasforest_pls'],
        'Plasgraph2': plasmid_classification['plasgraph2_pls'],
        'Platon': plasmid_classification['platon_pls'],
        'RFPlasmid': plasmid_classification['rfplasmid_pls']
    }
    
    return venn_data

# Main folder containing subfolders
main_folder_path = '/home/mrs27/projects/ctb-chauvec/PLASMIDS/DATA/Ecoli_Features'  # Replace with your actual main folder path

# Step 1: Process each subfolder and generate Venn diagrams
accumulated_venn_data = defaultdict(lambda: set())
subfolder_count = 0

for subfolder in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        feature_file_path = os.path.join(subfolder_path, 'features.tsv')
        if os.path.exists(feature_file_path):
            # Process the feature file
            venn_data = process_file(feature_file_path)
            subfolder_count += 1
            
            # Save individual Venn diagram
            venn(venn_data)
            plt.title(f'Plasmid Classification Venn Diagram for {subfolder}')
            output_file = os.path.join(subfolder_path, f'{subfolder}_venn_diagram.png')
            plt.savefig(output_file)
            plt.close()

            print(f"The Venn diagram for {subfolder} has been saved as {output_file}")
            
            # Accumulate Venn data
            for key in venn_data:
                accumulated_venn_data[key].update(venn_data[key])

# Step 2: Plot the average Venn diagram
if subfolder_count > 0:
    venn(accumulated_venn_data)
    plt.title('Cumulative Plasmid Classification Venn Diagram')

    # Save the cumulative Venn diagram
    cumulative_output_file = os.path.join(main_folder_path, 'cumulative_venn_diagram.png')
    plt.savefig(cumulative_output_file)
    plt.close()

    print(f"The cumulative Venn diagram has been saved as {cumulative_output_file}")
else:
    print("No feature files found in the subfolders.")
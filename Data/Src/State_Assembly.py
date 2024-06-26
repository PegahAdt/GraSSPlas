import os
import gzip
import matplotlib.pyplot as plt

def parse_fasta(fasta_path):
    """Parse a FASTA file to count contigs and calculate length distributions."""
    contig_lengths = []
    with gzip.open(fasta_path, 'rt') as file:  # Open in text mode for reading
        length = 0
        for line in file:
            if line.startswith('>'):
                if length > 0:
                    contig_lengths.append(length)
                    length = 0
            else:
                length += len(line.strip())
        if length > 0:  # Add the last contig if exists
            contig_lengths.append(length)
    return len(contig_lengths), contig_lengths

def parse_gfa(gfa_path):
    """Parse a GFA file to count nodes, edges, and calculate length distributions of nodes."""
    node_lengths = []
    edge_count = 0
    with gzip.open(gfa_path, 'rt') as file:
        for line in file:
            parts = line.strip().split('\t')
            if parts[0] == 'S':
                node_lengths.append(len(parts[2]))  # Assuming column 3 contains sequence
            elif parts[0] == 'L':
                edge_count += 1
    return len(node_lengths), edge_count, node_lengths

def plot_length_distribution(lengths, folder_path, title, filename):
    """Plot and save the length distribution of nodes or edges."""
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    stat_folder_path = os.path.join(folder_path, 'stat')  # Define the stat folder path
    if not os.path.exists(stat_folder_path):  # Check if the stat folder exists
        os.makedirs(stat_folder_path)  
    plt.savefig(os.path.join(stat_folder_path, filename))
    plt.close()

def save_analysis_data(stat_folder_path, analysis_data):
    """Save analysis data to a text file in the specified stat folder."""
    if not os.path.exists(stat_folder_path):  # Ensure the stat folder exists
        os.makedirs(stat_folder_path)
    with open(os.path.join(stat_folder_path, 'analysis_results.txt'), 'w') as analysis_file:
        analysis_file.write('\n'.join(analysis_data))

def analyze_folder(folder_path):
    """Traverse folders and perform analysis on FASTA and GFA files, plot length distributions, and save analysis data to a text file in a stat folder."""
    for root, dirs, files in os.walk(folder_path):
        analysis_data = []  # List to hold analysis data strings
        for file in files:
            if file.endswith('.fasta.gz'):
                contig_count, contig_lengths = parse_fasta(os.path.join(root, file))
                analysis_data.append(f'{file}: Contigs={contig_count}')
                plot_length_distribution(contig_lengths, root, 'Contig Length Distribution', f'{file}_contig_length_distribution.png')
            elif file.endswith('.gfa.gz'):
                node_count, edge_count, node_lengths = parse_gfa(os.path.join(root, file))
                analysis_data.append(f'{file}: Nodes={node_count}, Edges={edge_count}')
                plot_length_distribution(node_lengths, root, 'Node Length Distribution', f'{file}_node_length_distribution.png')
        
        # Call the new function to save analysis data to a text file in the stat folder within the current subfolder
        if analysis_data:
            stat_folder_path = os.path.join(root, 'stat')  # Define the stat folder path
            save_analysis_data(stat_folder_path, analysis_data)

# Example usage
folder_path = '/Users/mrsadeghian/Desktop/MrS/Research/GraSSPlas/Data/Ecoli_AssemblyGraph'
analyze_folder(folder_path)
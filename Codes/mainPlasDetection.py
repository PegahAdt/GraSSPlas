from toolsDetection import *
import torch
from torch_geometric.data import Data
import numpy as np
import argparse as ap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from upsetplot import UpSet

# Check torch version
print('torch: ', torch.__version__)

# Set this to 1 if we only have two classes: plasmid and not-plasmid
binaryClassification = 1

# Define command-line arguments
parser = ap.ArgumentParser()
parser.add_argument('--name', help='Folder containing reads and reference genomes (if available) in Data folder', default="shakya_1")
parser.add_argument('--p', type=int, help='p threshold to generate pseudo-labels', default=35)
parser.add_argument('--N_iter', type=int, help='Number of iterations', default=10)
parser.add_argument('--isSemiSupervised', type=int, help='Flag to use semi-supervised learning instead of self-supervised learning', default=0)
parser.add_argument('--noGNN', type=int, help='Flag to exclude GNN step or not', default=0)
parser.add_argument('--noRF', type=int, help='Flag to exclude RF step or not', default=1)
args = parser.parse_args()

# Initialize variables
name = args.name
N_tree = 100
hidden_channel = 16
embSize = 8
N_epoch_GNN = 2001
gnnLr = 0.001

# Main execution
print("Reading data:")
file_path = args.name
edges_list, features_list, labels_list, sample_numbers, GT = read_files_and_modify_contig_name(file_path)

# Infer the index to contig map
index_to_contig_map = pd.DataFrame({
    'index': features_list.index,
    'contig': features_list['contig']
})

# Create a dictionary for contig to index mapping
contig_to_index_map = dict(zip(index_to_contig_map['contig'], index_to_contig_map['index']))

# Generate the new edges object with indices
index_edges = [replace_contig_with_index(edge, contig_to_index_map) for edge in edges_list]
edge_indices = np.array([(edge[0], edge[1]) for edge in index_edges])

# Merge features and labels DataFrames
merged_data = pd.merge(features_list, labels_list, on='contig')

# Check if the length of sample_numbers matches the number of rows in merged_data
if len(merged_data) == len(sample_numbers):
    merged_data['sample'] = sample_numbers
else:
    raise ValueError("The length of sample_numbers does not match the number of rows in merged_data")

# Plot histograms for specific columns
fig, axs = plt.subplots(nrows=len(['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls', 'contig_length']), figsize=(8, 6 * len(['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls', 'contig_length'])))
for idx, col in enumerate(['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls', 'contig_length']):
    axs[idx].hist(merged_data[col], bins=10)  # Adjust the number of bins as needed
    axs[idx].set_title(col)
    axs[idx].set_xlabel('Score')
    axs[idx].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('score_frequency_plots.png')
plt.show()

print("Columns of merged_data before merging new features:")
print(merged_data.columns)

feature_list_for_avg = ['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls']

new_features_df = compute_new_features_from_edges(edge_indices, merged_data, feature_list_for_avg)
print("Columns of new_features_df before merging new features:")
print(new_features_df.columns)

print("Number of rows in new_features_df:", len(new_features_df))
print("Number of rows in merged_data:", len(merged_data))

# Merge original features with new features
merged_data = pd.concat([merged_data, new_features_df], axis=1)
print("Number of rows in final merged_data:", len(merged_data))
print("Columns of merged_data after merging new features:")
print(merged_data.columns)
print(merged_data)

# Normalize 'read_depth' and 'length' columns
max_length = merged_data['length'].max()
max_read_depth = merged_data['read_depth'].max()
merged_data['read_depth'] = merged_data['read_depth'] / max_read_depth
merged_data['length'] = merged_data['length'] / max_length

# Create features array
columns_to_drop = [col for col in merged_data.columns if col in labels_list.columns]
X_raw = merged_data.drop(columns=columns_to_drop)

print("Minimum values:")
print(merged_data.describe(include='all').loc['min'])

print("\nMaximum values:")
print(merged_data.describe(include='all').loc['max'])

# Exclude specific columns
X_raw = X_raw.drop(columns=["contig_length"])
print("X_raw:\n", X_raw.columns)
X_raw = X_raw.to_numpy()

# Prepare ground truth binary labels
y_binary_df = GT['Plasmid']
y_binary_df = pd.to_numeric(y_binary_df, errors='coerce')
y_binary_df = y_binary_df.dropna()
y_binary = y_binary_df.to_numpy(dtype=np.int64)

# Convert data to PyTorch tensors
num_features = X_raw.shape[1]
X_attr = torch.tensor(X_raw, dtype=torch.float)
edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
y_binary = torch.tensor(y_binary, dtype=torch.long)

# Form the graph data
data_main = Data(x=X_attr, edge_index=edge_index, y_binary=y_binary)
y_initial, train_mask = get_yinitial(labels_list, binaryClassification)
data_main.y_initial = torch.tensor(y_initial, dtype=torch.long)
data_main.train_mask = torch.tensor(train_mask, dtype=torch.bool)
merged_data['y_initial'] = y_initial
merged_data['train_mask'] = train_mask

# Compute statistics
Input_stats = compute_statistics(merged_data)

# Convert stats to DataFrame and save to TSV file
stats_df = stats_to_dataframe(Input_stats)
stats_df.to_csv('stats_output.tsv', sep='\t', index=False)

# Save various components to text files
np.savetxt('data_main_node_features.txt', data_main.x.numpy())
np.savetxt('data_main_edge_index.txt', data_main.edge_index.numpy())
np.savetxt('data_main_binary_labels.txt', data_main.y_binary.numpy())
np.savetxt('data_main_initial_labels.txt', data_main.y_initial.numpy())
np.savetxt('data_main_training_mask.txt', data_main.train_mask.numpy())

# Print configuration details
print("name:", args.name)
print("Testing threshold: ", args.p)
print("N_epoch_GNN:", N_epoch_GNN)
print("N_tree:", N_tree)
print("isSemiSupervised:", args.isSemiSupervised)
print("hidden_channel:", hidden_channel)
print("embSize:", embSize)
print("gnnLr:", gnnLr)
print("")

# Initialize performance metrics lists
Test_Acc_gnn = []
Test_Sen_gnn = []
Test_Pre_gnn = []
Test_tpr_gnn = []
Test_fpr_gnn = []
Test_f1_gnn = []
conf_matrixes_gnn = []

Test_Acc_rf = []
Test_Sen_rf = []
Test_Pre_rf = []
Test_tpr_rf = []
Test_fpr_rf = []
Test_f1_rf = []
conf_matrixes_rf = []
Test_Fi_rf = []

# Training and evaluation loop
for iter in range(args.N_iter):
    print("iter = ", iter)
    
    # Initialize GNN model
    model_GNN = GCN(in_channels=num_features, hidden_channels=hidden_channel, embSize=embSize, num_classes=2)
    X, y_true, y_pred_gnn, y_initial, confusion_matrix_gnn_iter, emb_all = embLearn(data_main, model_GNN, N_epoch_GNN, gnnLr, args.noGNN)
    
    # Calculate GNN performance metrics
    conf_gnn = confusion_matrix(y_true, y_pred_gnn)
    acc_gnn = np.trace(conf_gnn) / np.sum(conf_gnn)
    sen_gnn = conf_gnn[1, 1] / np.sum(conf_gnn[1, :]) if np.sum(conf_gnn[1, :]) != 0 else 0
    pre_gnn = conf_gnn[1, 1] / np.sum(conf_gnn[:, 1]) if np.sum(conf_gnn[:, 1]) != 0 else 0
    tpr_gnn = sen_gnn
    fpr_gnn = conf_gnn[0, 1] / np.sum(conf_gnn[0, :]) if np.sum(conf_gnn[0, :]) != 0 else 0
    f1_gnn = 2 * (pre_gnn * sen_gnn) / (pre_gnn + sen_gnn) if (pre_gnn + sen_gnn) != 0 else 0
    Test_Acc_gnn.append(acc_gnn)
    Test_Sen_gnn.append(sen_gnn)
    Test_Pre_gnn.append(pre_gnn)
    Test_tpr_gnn.append(tpr_gnn)
    Test_fpr_gnn.append(fpr_gnn)
    Test_f1_gnn.append(f1_gnn)
    conf_matrixes_gnn.append(conf_gnn)
    
    # Perform Random Forest learning if not excluded
    if args.noRF == 1:
        y_pred_rf = y_pred_gnn
        acc_rf = acc_gnn
        sen_rf = sen_gnn
        pre_rf = pre_gnn
        conf_rf = conf_gnn
        feature_importance_rf = np.zeros(X.shape[1])
    else:
        acc_rf, sen_rf, pre_rf, conf_rf, y_pred_rf, feature_importance_rf = rfLearn(X, y_initial, y_true, train_mask, N_tree)

    # Calculate RF performance metrics
    tpr_rf = sen_rf
    fpr_rf = conf_rf[0, 1] / np.sum(conf_rf[0, :]) if np.sum(conf_rf[0, :]) != 0 else 0
    f1_rf = 2 * (pre_rf * sen_rf) / (pre_rf + sen_rf) if (pre_rf + sen_rf) != 0 else 0
    Test_Acc_rf.append(acc_rf)
    Test_Sen_rf.append(sen_rf)
    Test_Pre_rf.append(pre_rf)
    Test_tpr_rf.append(tpr_rf)
    Test_fpr_rf.append(fpr_rf)
    Test_f1_rf.append(f1_rf)
    conf_matrixes_rf.append(conf_rf)
    Test_Fi_rf.append(feature_importance_rf)

    y_final = y_pred_rf



# Prepare tools DataFrame and apply threshold condition
tools = merged_data[['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls']]
tools = tools.applymap(lambda x: 1 if x > 0.5 else 0)
plot_venn_diagram(tools, ['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls'], 'upset_diagram_log.png')

tools['GrassPlas'] = y_final

# Merge tools with ground truth and sample data
merged_df = pd.merge(tools, GT, how='outer', left_index=True, right_index=True)
merged_df = pd.merge(merged_df, merged_data[['sample', 'contig_length']], left_index=True, right_index=True)

# Define the thresholds for contig length
thresholds = [0, 100, 1000]
unique_samples = merged_df['sample'].unique()

# Iterate over thresholds and compute metrics
for threshold in thresholds:
    filtered_df = merged_df[merged_df['contig_length'] >= threshold]
    tools = filtered_df[['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls', 'GrassPlas']]
    compute_and_save_metrics(tools, filtered_df, f'tools_analysis_threshold_{threshold}.txt')

# Save the output to a CSV file
output_data = pd.DataFrame({
    'index': index_to_contig_map['index'],
    'contig': index_to_contig_map['contig'],
    'X_raw': list(X_raw),
    'y_binary': y_binary.numpy(),
    'pseudo-label': y_initial,
    'train_mask': train_mask,
    'y_final': y_final,
    'ground_Truth': GT['Plasmid']
})

output_data.to_csv('real_output.csv', index=False)


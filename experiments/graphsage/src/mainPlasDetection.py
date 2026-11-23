from toolsDetection import *
import torch
print('torch: ', torch.__version__)
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse as ap
import pandas as pd
import os


# Parameters:
parser = ap.ArgumentParser()
parser.add_argument('--name', help='Folder containing reads and reference genomes (if available) in Data folder', default="shakya_1")
parser.add_argument('--p', type=int, help='p threshold to generate pseudo-labels', default=35)
parser.add_argument('--N_iter', type=int, help='Number of iteration', default=10)
parser.add_argument('--isSemiSupervised', type=int, help='Flag to use semi-supervised learning instead of self-supervised learning', default=0)
parser.add_argument('--noGNN', type=int, help='Flag to exclude GNN step or not', default=0)
parser.add_argument('--noRF', type=int, help='Flag to exclude RF step or not', default=1)
parser.add_argument('--output_dir', required=True, help='Directory to store output files')
args = parser.parse_args()


name = args.name
N_tree = 100
thresholds = args.p
hidden_channel = 16
embSize = 8
N_epoch_GNN = 2001
gnnLr = 0.001
output_dir=args.output_dir
length_threshold=100
"""# **Data loading:**"""

edges_list, combined_features_df, combined_labels_df, sample_numbers, ground_truth_indicator_df= process_files(name,length_threshold)

index_to_contig_map = pd.DataFrame({
    'index': combined_features_df.index,
    'contig': combined_features_df['contig']
})

# Create a dictionary for contig to index mapping
contig_to_index_map = dict(zip(index_to_contig_map['contig'], index_to_contig_map['index']))

# Generate the new edges object with indices
index_edges = [replace_contig_with_index(edge, contig_to_index_map) for edge in edges_list]
edge_indices = np.array([(edge[0], edge[1]) for edge in index_edges])

# Print the dimensions of each variable
print(f"Dimensions of edges_list: {len(edges_list)}")  # edges_list is a list, so we print its length
print(f"Dimensions of combined_features_df: {combined_features_df.shape}")
print(f"Dimensions of combined_labels_df: {combined_labels_df.shape}")
print(f"Dimensions of sample_numbers: {len(sample_numbers)}")  # sample_numbers is a list or array, so we print its length
print(f"Dimensions of ground_truth_indicator_df: {ground_truth_indicator_df.shape}")



output_file_path = os.path.join(args.output_dir, 'features.csv')
combined_features_df.to_csv(output_file_path, index=False)

# Save combined_labels_df
output_file_path_labels = os.path.join(args.output_dir, 'labels.csv')
combined_labels_df.to_csv(output_file_path_labels, index=False)

# Save ground_truth_indicator_df
output_file_path_gt = os.path.join(args.output_dir, 'ground_truth.csv')
ground_truth_indicator_df.to_csv(output_file_path_gt, index=False)


X_attr_df = combined_features_df.drop(columns=['contig','length',
       'read_depth', 'gc_content'])

num_features = X_attr_df.shape[1]

# Convert combined_features_df to a PyTorch tensor
X_attr = torch.tensor(X_attr_df.values, dtype=torch.float)

# Convert edge_indices to a PyTorch tensor in COO format
edge_index = torch.tensor(edge_indices.T, dtype=torch.long)

# Convert ground_truth_indicator_df to a PyTorch tensor
plasmid_column = ground_truth_indicator_df['Plasmid'].apply(pd.to_numeric, errors='coerce')

# Check for any NaNs or invalid values
if plasmid_column.isnull().any():
    raise ValueError("The 'Plasmid' column contains NaNs or invalid values. Please clean the data before conversion.")

# Convert the column to a numpy array
y_np = plasmid_column.values

# Convert the numpy array to a PyTorch tensor
y = torch.tensor(y_np, dtype=torch.long)

y_initial, train_mask = get_yinitial(combined_labels_df)

# Create the Data object
data_main = Data(x=X_attr, edge_index=edge_index,  y_binary=y)
data_main.y_initial = torch.tensor(y_initial, dtype=torch.long)
data_main.train_mask = torch.tensor(train_mask, dtype=torch.bool)




print(data_main)



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

if isinstance(y_final, torch.Tensor):
    y_final = y_final.numpy()

# Convert y_final to a DataFrame or Series
y_final_df = pd.DataFrame(y_final, columns=['y_final'])
num_ones = (y_final_df['y_final'] == 1).sum()
num_zeros = (y_final_df['y_final'] == 0).sum()

# Print the counts
print(f"Number of ones in y_final: {num_ones}")
print(f"Number of zeros in y_final: {num_zeros}"
)
tools = X_attr_df [['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls','mlplasmid_pls']]
tools = tools.applymap(lambda x: 1 if x > 0.5 else 0)
tools['GrassPlas'] = y_final_df

length_column = combined_features_df['length']
df_for_eval = pd.concat([tools, plasmid_column, length_column], axis=1)

output_file_path_gt = os.path.join(args.output_dir, 'y_final.csv')
y_final_df.to_csv(output_file_path_gt, index=False)

# Define the thresholds for contig length
thresholds = [0, 100, 1000]
# Iterate over thresholds and compute metrics
for threshold in thresholds:
    filtered_df = df_for_eval[df_for_eval['length'] >= threshold]
    filtered_tools = filtered_df[['plasforest_pls', 'plasgraph2_pls', 'platon_pls', 'rfplasmid_pls','mlplasmid_pls', 'GrassPlas']]
    metrics_file_path = os.path.join(output_dir, f'metrics_threshold_{threshold}.txt')
    compute_and_save_metrics(filtered_tools, filtered_df, metrics_file_path)

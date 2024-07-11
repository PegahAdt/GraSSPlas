import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from upsetplot import UpSet
import networkx as nx

# Define the GCN model using SAGEConv
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embSize, num_classes):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embSize)
        self.bn2 = torch.nn.BatchNorm1d(embSize)
        self.fc = torch.nn.Linear(embSize, num_classes)

    def forward(self, x, edge_index):
        x0 = self.conv1(x, edge_index)
        x0 = self.bn1(x0)
        x0 = F.relu(x0)
        x0 = self.conv2(x0, edge_index)
        x0 = self.bn2(x0)
        x0 = F.relu(x0)
        x1 = self.fc(x0)
        x1 = F.softmax(x1, dim=1)
        return x0, x1

# Function to compute the training mask
def compute_train_mask(row):
    return np.logical_or(row['plasmid'], row['chromosome'])

# Function to get initial labels and training mask
def get_yinitial(labels_list, binaryClassification):
    if binaryClassification == 1:
        if isinstance(labels_list, pd.DataFrame):
            y_initial = labels_list["plasmid"].values.astype(int)
            train_mask = labels_list.apply(compute_train_mask, axis=1).values
        else:
            raise ValueError("labels_list must be a DataFrame")
    else:
        raise NotImplementedError("Multiclass classification not implemented yet")
    return y_initial, train_mask

# Function to learn embeddings using GNN
def embLearn(data_in, model_GNN, N_epoch_GNN, gnnLr, noGNN):
    print("Learning embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    x_all, edge_index_all, y_all, y_initial, train_mask = (
        data_in.x.to(device),
        data_in.edge_index.to(device),
        data_in.y_binary.to(device),
        data_in.y_initial.to(device),
        data_in.train_mask.to(device),
    )
    model_GNN = model_GNN.to(device)

    optimizer_GNN = torch.optim.Adam(model_GNN.parameters(), lr=gnnLr)
    criterion_GNN = torch.nn.CrossEntropyLoss()

    N_epoch_GNN_set = 1 if noGNN else N_epoch_GNN
    model_GNN.train()
    for epoch in range(N_epoch_GNN_set):
        optimizer_GNN.zero_grad()
        emb_, out_GNN = model_GNN(x_all, edge_index_all)
        loss = criterion_GNN(out_GNN[train_mask], y_initial[train_mask])
        loss.backward()
        optimizer_GNN.step()
        epoch_pred = out_GNN[train_mask].argmax(dim=1).cpu().numpy()
        epoch_real = y_initial[train_mask].cpu().numpy()
        loss_epoch = loss.item()

        if epoch % 100 == 0:
            acc = accuracy_score(epoch_real, epoch_pred)
            rec = recall_score(epoch_real, epoch_pred, average='macro')
            print('Epoch: {:03d}, Loss: {:.5f}, Acc: {:.5f}, Rec: {:.5f}'.format(epoch, loss_epoch, acc, rec))

    model_GNN.eval()
    with torch.no_grad():
        emb_GNN_all, out = model_GNN(x_all, edge_index_all)
        y_pred = out.argmax(dim=1).cpu().numpy()
        y_true = y_all.cpu().numpy()

    confusion_matrix_gnn = confusion_matrix(y_true, y_pred)
    emb_GNN_all = emb_GNN_all.cpu().numpy()
    x_features = x_all.cpu().numpy()
    y_initial = y_initial.cpu().numpy()

    X = x_features if noGNN else np.concatenate((x_features, emb_GNN_all), axis=1)
    return X, y_true, y_pred, y_initial, confusion_matrix_gnn, emb_GNN_all

# Function to train and evaluate Random Forest classifier
def rfLearn(X, y_initial, y_true, train_mask, N_tree):
    X_train = X[train_mask]
    y_train = y_initial[train_mask]
    rf = RandomForestClassifier(n_estimators=N_tree, n_jobs=1)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X)
    conf_rf = confusion_matrix(y_true, y_pred_rf)
    acc_rf = np.trace(conf_rf) / np.sum(conf_rf)
    sen_rf = conf_rf[1, 1] / np.sum(conf_rf[1, :]) if np.sum(conf_rf[1, :]) != 0 else 0
    pre_rf = conf_rf[1, 1] / np.sum(conf_rf[:, 1]) if np.sum(conf_rf[:, 1]) != 0 else 0

    return acc_rf, sen_rf, pre_rf, conf_rf, y_pred_rf, rf.feature_importances_

# Function to read GFA files
def read_gfa(file_path, line_number):
    edges = []
    contig_lengths = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('L'):
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    from_node = f"{line_number}_{parts[1]}"
                    to_node = f"{line_number}_{parts[3]}"
                    edges.append((from_node, to_node))
            elif line.startswith('S'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    contig_name = f"{line_number}_{parts[1]}"
                    contig_length = len(parts[2])
                    contig_lengths[contig_name] = contig_length
    return edges, contig_lengths

# Function to read files and modify contig names
def read_files_and_modify_contig_name(file_path):
    edges_list = []
    features_list = []
    labels_list = []
    ground_truth_list = []
    sample_numbers = []
    contig_lengths_all = {}

    with open(file_path, 'r') as file:
        next(file)
        for i, line in enumerate(file, start=1):
            print("Line contents:", line.strip())
            if line.strip():
                sample_number, graph_path, feature_path, label_path, ground_truth_path = line.strip().split(',')

                edges, contig_lengths = read_gfa(graph_path, i)
                edges_list.extend(edges)
                contig_lengths_all.update(contig_lengths)

                features_df = pd.read_csv(feature_path, sep='\t')
                features_df['contig'] = features_df['contig'].apply(lambda x: f"{i}_{x}")
                features_df['contig_length'] = features_df['contig'].map(contig_lengths)
                features_list.append(features_df)
                labels_df = pd.read_csv(label_path, sep='\t')
                labels_df['contig'] = labels_df['contig'].apply(lambda x: f"{i}_{x}")
                labels_list.append(labels_df)

                ground_truth_df_file = pd.read_csv(ground_truth_path, sep='\t')
                if 'contig' in ground_truth_df_file.columns:
                    ground_truth_df = ground_truth_df_file[['contig']]
                    ground_truth_df = ground_truth_df.dropna(subset=['contig'])
                    ground_truth_df['contig'] = ground_truth_df['contig'].apply(lambda x: f"{i}_{int(x)}")
                    ground_truth_list.append(ground_truth_df)
                else:
                    raise KeyError("'contig' column not found in the DataFrame")

                sample_numbers.extend([i] * len(features_df))

    all_features_df = pd.concat(features_list, ignore_index=True)
    all_labels_df = pd.concat(labels_list, ignore_index=True)
    all_ground_truth_df = pd.concat(ground_truth_list, ignore_index=True)

    ground_truth_set = set(all_ground_truth_df['contig'])
    all_contigs = all_features_df['contig']
    ground_truth_indicator_df = pd.DataFrame({
        'contig': all_contigs,
        'Plasmid': all_contigs.apply(lambda x: 1 if x in ground_truth_set else 0)
    })

    features_list_size = all_features_df.shape

    return edges_list, all_features_df, all_labels_df, sample_numbers, ground_truth_indicator_df

# Function to replace contig names with indices in each edge
def replace_contig_with_index(edge, contig_to_index_map):
    return [contig_to_index_map[node] for node in edge]

# Function to compute statistics for contig data
def compute_statistics(data, length_thresholds=[0, 100, 1000]):
    stats = defaultdict(lambda: defaultdict(int))
    samples = data['sample'].unique()
    stats['total_samples'] = len(samples)
    print("Number Of Samples:", stats['total_samples'])

    for sample in samples:
        sample_data = data[data['sample'] == sample]
        stats['samples'][sample] = defaultdict(int)
        stats['samples'][sample]['total_contigs'] = len(sample_data)
        stats['samples'][sample]['total_length'] = sample_data['contig_length'].sum()

        for threshold in length_thresholds:
            filtered_data = sample_data[sample_data['contig_length'] >= threshold]
            stats['samples'][sample][f'contigs_>=_{threshold}'] = len(filtered_data)
            stats['samples'][sample][f'length_>=_{threshold}'] = filtered_data['contig_length'].sum()
            stats['samples'][sample][f'contigs_with_label_>=_{threshold}'] = len(filtered_data[filtered_data['train_mask'] == 1])

    for threshold in length_thresholds:
        filtered_data = data[data['contig_length'] >= threshold]
        stats['total'][f'contigs_>=_{threshold}'] = len(filtered_data)
        stats['total'][f'length_>=_{threshold}'] = filtered_data['contig_length'].sum()
        stats['total'][f'contigs_with_label_>=_{threshold}'] = len(filtered_data[filtered_data['train_mask'] == 1])

    return stats

# Function to convert statistics to DataFrame
def stats_to_dataframe(stats):
    rows = []

    for sample, sample_stats in stats['samples'].items():
        row = {'sample': sample}
        row.update(sample_stats)
        rows.append(row)

    global_stats = {'sample': 'total'}
    global_stats.update(stats['total'])
    rows.append(global_stats)

    return pd.DataFrame(rows)

# Function to compute and save metrics
def compute_and_save_metrics(tools_df, ground_truth_df, filename):
    metrics = {}
    ground_truth_df = ground_truth_df.loc[tools_df.index]

    for column in tools_df.columns:
        y_pred = tools_df[column]
        y_true = ground_truth_df['Plasmid']

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics[column] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            }
        }

    # Calculate intersection and union metrics
    y_true = ground_truth_df['Plasmid']
    
    intersection_pred = tools_df.all(axis=1).astype(int)
    union_pred = tools_df.any(axis=1).astype(int)
    
    intersection_precision = precision_score(y_true, intersection_pred)
    intersection_recall = recall_score(y_true, intersection_pred)
    intersection_f1 = f1_score(y_true, intersection_pred)
    intersection_tn, intersection_fp, intersection_fn, intersection_tp = confusion_matrix(y_true, intersection_pred).ravel()
    
    union_precision = precision_score(y_true, union_pred)
    union_recall = recall_score(y_true, union_pred)
    union_f1 = f1_score(y_true, union_pred)
    union_tn, union_fp, union_fn, union_tp = confusion_matrix(y_true, union_pred).ravel()
    
    metrics['intersection'] = {
        'precision': intersection_precision,
        'recall': intersection_recall,
        'f1': intersection_f1,
        'confusion_matrix': {
            'TP': intersection_tp,
            'FP': intersection_fp,
            'FN': intersection_fn,
            'TN': intersection_tn
        }
    }
    
    metrics['union'] = {
        'precision': union_precision,
        'recall': union_recall,
        'f1': union_f1,
        'confusion_matrix': {
            'TP': union_tp,
            'FP': union_fp,
            'FN': union_fn,
            'TN': union_tn
        }
    }

    with open(filename, 'w') as file:
        for tool, metric in metrics.items():
            file.write(f"Tool: {tool}\n")
            file.write(f"Precision: {metric['precision']}\n")
            file.write(f"Recall: {metric['recall']}\n")
            file.write(f"F1 Score: {metric['f1']}\n")
            file.write(f"Confusion Matrix: TP={metric['confusion_matrix']['TP']}, FP={metric['confusion_matrix']['FP']}, FN={metric['confusion_matrix']['FN']}, TN={metric['confusion_matrix']['TN']}\n")
            file.write("\n")

# Function to plot UpSet diagram
def plot_venn_diagram(df, tool_columns, output_filename):
    df_upset = df.astype(bool).groupby(df.columns.tolist()).size()
    upset = UpSet(df_upset)
    fig = upset.plot()
    intersection_ax = fig['intersections']
    intersection_ax.set_yscale('log')
    plt.savefig(output_filename)

# Function to compute new features from edges
def compute_new_features_from_edges(edge_indices, features_df, feature_list):
    G = nx.Graph()
    G.add_edges_from(edge_indices)
    new_features_df = pd.DataFrame(index=features_df.index)

    for feature in feature_list:
        new_feature_col = f"{feature}_avg"
        new_features_df[new_feature_col] = 0.0

        for node in G.nodes:
            neighbors = list(G.neighbors(node))

            if neighbors:
                neighbor_values = features_df.loc[neighbors, feature]
                avg_value = neighbor_values.mean()
            else:
                avg_value = features_df.loc[node, feature]

            new_features_df.loc[node, new_feature_col] = avg_value

    return new_features_df


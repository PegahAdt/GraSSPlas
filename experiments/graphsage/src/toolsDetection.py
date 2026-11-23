import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
#from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.loader import NodeLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embSize, num_classes):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x0 = self.conv1(x, edge_index)  # Apply the graph convolutional layer
        x0 = self.bn1(x0)               # Apply batch normalization
        x0 = F.relu(x0)                 # Apply ReLU activation
        
        x1 = self.fc(x0)                # Apply the fully connected layer
        x1 = F.softmax(x1, dim=1)       # Apply softmax to get probabilities

        return x0, x1  # Return both x0 and x1




def embLearn(data_in, model_GNN, N_epoch_GNN, gnnLr, noGNN):
    print("learning embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    x_all, edge_index_all, y_all, y_initial, train_mask = (
    data_in.x.to(device),
    data_in.edge_index.to(device),
    data_in.y_binary.to(device),
    data_in.y_initial.to(device),
    data_in.train_mask.to(device),)
    
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
        epoch_pred = out_GNN[train_mask].argmax(dim=1).cpu().numpy()  # Move predictions back to CPU
        epoch_real = y_initial[train_mask].cpu().numpy()  # Move labels back to CPU
        loss_epoch = loss.item()
        
        if epoch % 100 == 0:
            acc = accuracy_score(epoch_real, epoch_pred)
            rec = recall_score(epoch_real, epoch_pred, average='macro')
            print('y_initial: Epoch: {:03d}, loss: {:.5f}, Acc: {:.5f}, Rec: {:.5f}'.format(epoch, loss_epoch, acc, rec))

    model_GNN.eval()
    with torch.no_grad():
        emb_GNN_all, out = model_GNN(x_all, edge_index_all)
        y_pred = out.argmax(dim=1).cpu().numpy()  # Move predictions back to CPU
        y_true = y_all.cpu().numpy()  # Move labels back to CPU

    confusion_matrix_gnn = confusion_matrix(y_true, y_pred)  # Convert predictions to CPU for the confusion matrix
    emb_GNN_all = emb_GNN_all.cpu().numpy()  # Move embeddings back to CPU

    # Create X_train and X_test by combining node features and GNN embeddings
    x_features = x_all.cpu().numpy()
    y_initial = y_initial.cpu().numpy()

    if noGNN:
        X = x_features
    else:
        X = np.concatenate((x_features, emb_GNN_all), axis=1)


    return X, y_true, y_pred, y_initial, confusion_matrix_gnn, emb_GNN_all

def rfLearn(X, y_initial, y_true, train_mask, N_tree):
    X_train = X[train_mask]
    y_train = y_initial[train_mask]
    rf = RandomForestClassifier(n_estimators=N_tree, n_jobs = 30)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X)
    conf_rf = confusion_matrix(y_true, y_pred_rf)
    acc_rf = np.trace(conf_rf) / np.sum(conf_rf) 
    sen_rf = conf_rf[1,1] / np.sum(conf_rf[1,:]) if np.sum(conf_rf[1,:])!=0 else 0
    pre_rf = conf_rf[1,1] / np.sum(conf_rf[:,1]) if np.sum(conf_rf[:,1])!=0 else 0
    
    return acc_rf, sen_rf, pre_rf, conf_rf, y_pred_rf, rf.feature_importances_

def parse_gfa_file(file_path, line_number, length_threshold):
    """Parse a GFA file to extract edges and filter contigs by length.

    Args:
        file_path (str): Path to the GFA file.
        line_number (int): Line number to prepend to contig names for uniqueness.
        length_threshold (int): Minimum length of contigs to keep.

    Returns:
        list: List of edges as tuples (from_node, to_node).
        dict: Dictionary of contig lengths.
        set: Set of valid contigs (above the length threshold).
    """
    edges = []
    contig_lengths = {}
    valid_contigs = set()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if line.startswith('L') and len(parts) >= 5:
                from_node = f"{line_number}_{parts[1]}"
                to_node = f"{line_number}_{parts[3]}"
                if from_node in valid_contigs and to_node in valid_contigs:
                    edges.append((from_node, to_node))
            elif line.startswith('S') and len(parts) >= 3:
                contig_name = f"{line_number}_{parts[1]}"
                contig_length = len(parts[2])
                contig_lengths[contig_name] = contig_length
                if contig_length >= length_threshold:
                    valid_contigs.add(contig_name)

    return edges, contig_lengths, valid_contigs

def modify_contig_names(df, line_number):
    """Modify contig names in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'contig' column.
        line_number (int): Line number to prepend to contig names for uniqueness.

    Returns:
        pd.DataFrame: Updated DataFrame with modified contig names.
    """
    df['contig'] = df['contig'].apply(lambda x: f"{line_number}_{x}")
    return df

def process_files(file_list_path, length_threshold):
    """Process files listed in the provided file path and return combined results.

    Args:
        file_list_path (str): Path to the file containing a list of file paths.
        length_threshold (int): Minimum length of contigs to keep.

    Returns:
        tuple: Tuple containing combined edges, features, labels, sample numbers,
               and ground truth indicators.
    """
    edges_list = []
    features_list = []
    labels_list = []
    ground_truth_list = []
    sample_numbers = []

    with open(file_list_path, 'r') as file:
        next(file)  # Skip header line
        for i, line in enumerate(file, start=1):
            sample_number, graph_path, feature_path, label_path, ground_truth_path = line.strip().split(',')

            edges, contig_lengths, valid_contigs = parse_gfa_file(graph_path, i, length_threshold)
            edges_list.extend(edges)

            features_df = pd.read_csv(feature_path, sep='\t')
            features_df = modify_contig_names(features_df, i)
            features_df = features_df[features_df['contig'].isin(valid_contigs)]
            features_list.append(features_df)

            labels_df = pd.read_csv(label_path, sep='\t')
            labels_df['contig'] = labels_df['contig'].apply(lambda x: f"{i}_{x}")
            labels_df = labels_df[labels_df['contig'].isin(valid_contigs)]
            labels_list.append(labels_df)

            ground_truth_df = read_ground_truth(ground_truth_path, i)
            ground_truth_df = ground_truth_df[ground_truth_df['contig'].isin(valid_contigs)]
            ground_truth_list.append(ground_truth_df)

            sample_numbers.extend([i] * len(features_df))

    combined_features_df = pd.concat(features_list, ignore_index=True)
    combined_labels_df = pd.concat(labels_list, ignore_index=True)
    combined_ground_truth_df = pd.concat(ground_truth_list, ignore_index=True)

    ground_truth_set = set(combined_ground_truth_df['contig'])
    ground_truth_indicator_df = create_ground_truth_indicator(combined_features_df, ground_truth_set)

    return edges_list, combined_features_df, combined_labels_df, sample_numbers, ground_truth_indicator_df

def read_ground_truth(ground_truth_path, line_number):
    """Read and process ground truth data from a file.

    Args:
        ground_truth_path (str): Path to the ground truth file.
        line_number (int): Line number to prepend to contig names for uniqueness.

    Returns:
        pd.DataFrame: Processed ground truth DataFrame.
    """
    ground_truth_df = pd.read_csv(ground_truth_path, sep='\t')
    if 'contig' in ground_truth_df.columns:
        ground_truth_df = ground_truth_df[['contig']].dropna()
        ground_truth_df['contig'] = ground_truth_df['contig'].apply(lambda x: f"{line_number}_{int(x)}")
        return ground_truth_df
    else:
        raise KeyError("'contig' column not found in the ground truth file.")

def create_ground_truth_indicator(features_df, ground_truth_set):
    """Create a DataFrame indicating whether each contig is a plasmid.

    Args:
        features_df (pd.DataFrame): DataFrame containing contig information.
        ground_truth_set (set): Set of contigs identified as plasmids.

    Returns:
        pd.DataFrame: DataFrame with contig and Plasmid indicator columns.
    """
    return pd.DataFrame({
        'contig': features_df['contig'],
        'Plasmid': features_df['contig'].apply(lambda x: 1 if x in ground_truth_set else 0)
    })

def replace_contig_with_index(edge, contig_to_index_map):
    """Replace contig names in an edge with corresponding indices.

    Args:
        edge (tuple): Tuple containing the contig names.
        contig_to_index_map (dict): Dictionary mapping contig names to indices.

    Returns:
        tuple: Tuple containing the contig indices.
    """
    return tuple(contig_to_index_map[node] for node in edge)

def compute_train_mask(row):
    """Compute the training mask based on the labels."""
    return np.logical_xor(row['plasmid'], row['chromosome'])

def get_yinitial(labels_list):
    """Get initial labels and training mask for training."""
    if isinstance(labels_list, pd.DataFrame):
        y_initial = labels_list["plasmid"].values.astype(int)
        train_mask = labels_list.apply(compute_train_mask, axis=1).values
    else:
        raise ValueError("labels_list must be a DataFrame")
   
    return y_initial, train_mask

def save_to_tsv(data, file_name):
    """Save data to a TSV file."""
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_name, sep='\t', index=False)
    elif isinstance(data, list):
        # Assuming the list is either a list of tuples or a list of scalars
        with open(file_name, 'w', newline='') as file:
            for item in data:
                if isinstance(item, tuple):
                    file.write('\t'.join(map(str, item)) + '\n')
                else:
                    file.write(str(item) + '\n')

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


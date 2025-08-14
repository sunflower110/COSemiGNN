import math
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def print_layer_parameters(model):
    total_params = 0
    print("Layer parameter details:")
    print("-" * 60)
    print(f"{'Layer Name':<20} {'Layer Type':<30} {'Parameter Count':<15}")
    print("-" * 60)

    for name, param in model.named_parameters():
        param_count = param.numel()
        # Extract layer name (removing .weight/.bias suffixes)
        layer_name = name.rsplit('.', 1)[0]
        layer_type = str(param.shape)
        print(f"{layer_name:<20} {layer_type:<30} {param_count:<15}")
        total_params += param_count

    print("-" * 60)
    print(f"Total model parameters: {total_params}")
    return total_params


def remove_extreme_values(df, n=5):
    indices_to_remove = set()

    for column in df.columns[2:-1]:
        unique_count = df[column].nunique()

        indices_to_remove.update(df.nlargest(n, column).index)
        indices_to_remove.update(df.nsmallest(n, column).index)

    filtered_df = df.drop(index=indices_to_remove)
    return filtered_df


def csr_to_torch_sparse(csr):
    values = csr.data
    indices = np.vstack((csr.row, csr.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = csr.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def create_adj(features_df: pd.DataFrame, edge_df=None):
    edge_df = edge_df

    if not features_df.empty:
        edge_df = edge_df[edge_df['txId1'].isin(features_df["txId"]) & edge_df['txId2'].isin(features_df["txId"])]

    edges = [tuple(x) for x in edge_df.values]

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(features_df.values[:, 0])
    G.add_edges_from(edges)

    nodes = list(G.nodes())
    nodes.sort()

    node_index = {node: idx for idx, node in enumerate(nodes)}

    num_nodes = len(nodes)
    row_indices = []
    col_indices = []

    for edge in edges:
        u, v = edge
        row_indices.append(node_index[u])
        col_indices.append(node_index[v])
        row_indices.append(node_index[v])
        col_indices.append(node_index[u])  # Undirected graph

    indices = torch.LongTensor([row_indices, col_indices])
    values = torch.FloatTensor(np.ones(len(row_indices)))
    shape = torch.Size([num_nodes, num_nodes])

    adj_matrix_torch = torch.sparse_coo_tensor(indices, values, shape)
    return indices, adj_matrix_torch


def centering_matrix(n):
    h = torch.eye(n) / n
    return h


def myplot(x, y1, y2=None, label1='label', label2='label', title='Result'):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # Display negative signs correctly
    plt.plot(x, y1, 'r-o', label=label1)  # Red solid line with circle markers
    if y2 is not None:
        plt.plot(x, y2, 'b-o', label=label2)

    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel('Time Slice')
    plt.ylabel('Anomaly Transaction F1 Score')
    plt.legend()
    plt.show()
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def load_ori(dataset_name):
    dataset = Planetoid(root='data', name=dataset_name)
    data = dataset[0]
    adj = to_scipy_sparse_matrix(data.edge_index)
    G = nx.from_scipy_sparse_array(adj)
    return G

def load_adjacency_matrix(file_path):
    adj = torch.load(file_path).cpu()
    return adj

def adjacency_to_graph(adj):
    G = nx.from_numpy_array(adj.numpy())
    return G

def degree_distribution(graph):
    degrees = [degree for node, degree in graph.degree(weight='weight')]
    hist, bins = np.histogram(degrees, bins=range(int(max(degrees)+2)), density=True)
    return hist, bins[:-1]

def plot_degree_distribution(dataset_name, file_paths, labels, colors):
    plt.figure(figsize=(12, 6))
    
    for file_path, label, color in zip(file_paths, labels, colors):
        adj = load_adjacency_matrix(file_path)
        G = adjacency_to_graph(adj)
        hist, bins = degree_distribution(G)
        
        plt.bar(bins, hist, width=0.8, color=color, alpha=0.6, label=label)
    
    G_ori = load_ori(dataset_name)
    hist_ori, bins_ori = degree_distribution(G_ori)
    plt.bar(bins_ori, hist_ori, width=0.8, color='y', alpha=0.6, label='Original Graph')
    
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.legend()
    plt.savefig('deg.pdf')

labels = ['DosCond', 'GCond', 'SGDD']
colors = ['b', 'r', 'g']
dataset_name =  'cora'
file_paths = [
    f'save/DosCond/adj_{dataset_name}_1.0_1.pt',
    f'save/GCond/adj_{dataset_name}_1.0_1.pt',
    f'save/SGDD/adj_{dataset_name}_1.0_1.pt'
]

plot_degree_distribution(dataset_name, file_paths, labels, colors)

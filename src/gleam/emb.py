import pandas as pd
import numpy as np
import torch
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec
from typing import Dict, List, Union, Optional

# ==========================================
# 1. Custom Weighted MetaPath2Vec
# ==========================================

# Inspired and modified based on https://github.com/morteza/weighted-metapath2vec.

class WeightedMetaPath2Vec(MetaPath2Vec):
    def __init__(self, edge_index_dict: Dict[str, torch.Tensor],
                 edge_weight_dict: Dict[str, torch.Tensor],
                 embedding_dim: int, metapath: List[str],
                 walk_length: int, context_size: int,
                 walks_per_node: int = 1, num_negative_samples: int = 1,
                 num_nodes_dict: Dict[str, int] = None,
                 sparse: bool = False):
        
        super().__init__(edge_index_dict, embedding_dim, metapath,
                         walk_length, context_size, walks_per_node,
                         num_negative_samples, num_nodes_dict, sparse)
        
        self.edge_weight_dict = edge_weight_dict

    def sample(self, start: torch.Tensor, metapath: List[str]) -> torch.Tensor:
        # Note: This Python-based sampling loop may be slower than C++ implementations
        # for very large graphs, but it supports the custom weights provided.
        node = start
        path = [node]
        for i, edge_type in enumerate(metapath):
            edge_index = self.edge_index_dict[edge_type]
            edge_weight = self.edge_weight_dict[edge_type]
            
            # Normalize weights for each start node
            # Note: For efficiency in batch processing, this logic assumes 'node' 
            # aligns with edge_index structure. If 'node' is a batch, this loop
            # might need to be vectorized further for speed.
            unique_start, inverse_index = torch.unique(edge_index[0], return_inverse=True)
            weight_sum = torch.zeros(unique_start.size(0), device=edge_weight.device)
            weight_sum.scatter_add_(0, inverse_index, edge_weight)
            normalized_weights = edge_weight / weight_sum[inverse_index]
            
            # Sample next nodes based on normalized weights
            row, col = edge_index
            
            # Masking for the current node (simplified for single-node logic)
            # For batch usage, this would require broadcasting or loop modification.
            mask = row == node 
            weights = normalized_weights[mask]
            next_node = col[mask]

            if weights.sum() > 0:
                node = next_node[torch.multinomial(weights, 1)]
            else:
                node = torch.tensor([node], device=node.device)
            path.append(node)

        return torch.cat(path)

def weighted_metapath2vec(data: Union[HeteroData, Dict[str, torch.Tensor]],
                          embedding_dim: int, metapath: List[str],
                          walk_length: int, context_size: int,
                          walks_per_node: int = 1, num_negative_samples: int = 1,
                          sparse: bool = False) -> WeightedMetaPath2Vec:
    
    if isinstance(data, HeteroData):
        edge_index_dict = {edge_type: data[edge_type].edge_index 
                           for edge_type in data.edge_types}
        edge_weight_dict = {edge_type: data[edge_type].edge_attr 
                            for edge_type in data.edge_types 
                            if hasattr(data[edge_type], 'edge_attr')}
        num_nodes_dict = {node_type: data[node_type].num_nodes 
                          for node_type in data.node_types}
    else:
        edge_index_dict = data
        edge_weight_dict = {k: torch.ones(v.size(1)) for k, v in data.items()}
        num_nodes_dict = None

    return WeightedMetaPath2Vec(
        edge_index_dict, edge_weight_dict, embedding_dim, metapath,
        walk_length, context_size, walks_per_node, num_negative_samples,
        num_nodes_dict, sparse
    )

# ==========================================
# 2. Missing Graph Helper Functions
# ==========================================

def create_knn_graph(data_df, k=10):
    """
    Constructs a kNN graph from embedding data (DataFrame).
    Returns edge_index and edge_attr (weights).
    """
    # Fit kNN
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(data_df.values)
    distances, indices = nbrs.kneighbors(data_df.values)
    
    # Create source and target nodes
    n_nodes = data_df.shape[0]
    source_nodes = np.repeat(np.arange(n_nodes), k)
    target_nodes = indices.flatten()
    
    # Flatten distances for weights
    edge_weights = distances.flatten()
    
    # Convert to tensors
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_attr

def process_bipartite_graph(graph_df):
    """
    Converts a bipartite adjacency DataFrame (rows=RNA, cols=ATAC)
    into PyG-compatible edge_index and edge_attr.
    """
    # Convert DataFrame to sparse coordinate format (row, col, value)
    if hasattr(graph_df, "values"):
        matrix = graph_df.values
    else:
        matrix = graph_df
        
    rows, cols = np.where(matrix > 0)
    values = matrix[rows, cols]
    
    # Convert to tensors
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(values, dtype=torch.float)
    
    return edge_index, edge_attr

def create_adjacency_matrix(indices_rna, indices_atac):
    """
    Constructs a binary adjacency matrix representing connections between RNA cells and ATAC cells.
    """
    n_rna = indices_atac.shape[0]
    n_atac = indices_rna.shape[0]
    adj_matrix = np.zeros((n_rna, n_atac), dtype=int)

    for i, neighbors in enumerate(indices_rna):
        for atac_idx in neighbors:
            adj_matrix[atac_idx, i] = 1

    for i, neighbors in enumerate(indices_atac):
        for rna_idx in neighbors:
            adj_matrix[i, rna_idx] = 1

    return adj_matrix

def perform_pca(data, n_components=50):
    """Standard PCA on DataFrame."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    return pd.DataFrame(data_pca, index=data.index, columns=[f'PC_{i+1}' for i in range(n_components)])

def perform_pca_scanpy(adata, n_components=50):
    """Standard PCA on AnnData."""
    adata_pca = adata.copy()
    sc.pp.scale(adata_pca, max_value=10)
    sc.tl.pca(adata_pca, n_comps=n_components)
    return pd.DataFrame(adata_pca.obsm['X_pca'], index=adata_pca.obs_names, columns=[f'PC_{i+1}' for i in range(n_components)])

# ==========================================
# 3. Integration & Refinement Logic
# ==========================================

def integrate_data_with_metapath2vec(gene_expression, peak_levels, graph, dimensions=64, walk_length=6, num_walks=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Construct KNN graphs
    edge_index_gene, edge_attr_gene = create_knn_graph(gene_expression)
    edge_attr_gene[edge_attr_gene > 0] = 1 
    
    edge_index_peak, edge_attr_peak = create_knn_graph(peak_levels)
    edge_attr_peak[edge_attr_peak > 0] = 1 
    
    # 2. Process bipartite graph
    edge_index_bipartite, edge_attr_bipartite = process_bipartite_graph(graph)
    
    num_gene_nodes = gene_expression.shape[0]
    num_peak_nodes = peak_levels.shape[0]
    
    # 3. Build HeteroData
    data = HeteroData()
    data['gene'].node_id = torch.arange(num_gene_nodes)
    data['peak'].node_id = torch.arange(num_peak_nodes)
    data['gene'].num_nodes = num_gene_nodes
    data['peak'].num_nodes = num_peak_nodes
    
    data['gene', 'sim_1', 'gene'].edge_index = edge_index_gene
    data['gene', 'sim_1', 'gene'].edge_attr = edge_attr_gene
    
    data['peak', 'sim_2', 'peak'].edge_index = edge_index_peak
    data['peak', 'sim_2', 'peak'].edge_attr = edge_attr_peak
    
    data['gene', 'sim_3', 'peak'].edge_index = edge_index_bipartite
    data['gene', 'sim_3', 'peak'].edge_attr = edge_attr_bipartite
    
    data['peak', 'sim_4', 'gene'].edge_index = edge_index_bipartite.flip(0)
    data['peak', 'sim_4', 'gene'].edge_attr = edge_attr_bipartite
    
    # 4. Define Metapaths
    metapath = [
        ('gene', 'sim_1', 'gene'),
        ('gene', 'sim_3', 'peak'),
        ('peak', 'sim_2', 'peak'),
        ('peak', 'sim_4', 'gene')
    ]
    
    # 5. Initialize Custom Weighted Model
    model = weighted_metapath2vec(
        data,
        embedding_dim=dimensions,
        metapath=metapath,
        walk_length=walk_length,
        context_size=5,
        walks_per_node=num_walks,
        num_negative_samples=5,
        sparse=True
    ).to(device)
    
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
            
    # Training Loop
    for epoch in range(1, 20):
        model.train()
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        gene_embeddings = model('gene').cpu().numpy()
        peak_embeddings = model('peak').cpu().numpy()
    
    return gene_embeddings, peak_embeddings

def refine_graph_and_prepare_data_scanpy(adata_rna, adata_atac, initial_pca, 
                                         initial_k=10, refine_k=6, 
                                         pca_components=50):
    # --- Step 1: Initial Graph ---
    X_ref_initial = initial_pca[0:adata_rna.n_obs].values
    X_query_initial = initial_pca[adata_rna.n_obs:(adata_rna.n_obs + adata_atac.n_obs)].values
    
    nn = NearestNeighbors(n_neighbors=initial_k, metric='euclidean')
    nn.fit(X_ref_initial)
    _, indices_rna = nn.kneighbors(X_query_initial)
    
    nn = NearestNeighbors(n_neighbors=initial_k, metric='euclidean')
    nn.fit(X_query_initial) 
    _, indices_atac = nn.kneighbors(X_ref_initial)

    adj_matrix = pd.DataFrame(create_adjacency_matrix(indices_rna, indices_atac))
    adj_matrix.index = adata_rna.obs_names
    adj_matrix.columns = adata_atac.obs_names
    rna_graph = adj_matrix.copy()

    common_indexes = set(rna_graph.index).intersection(set(rna_graph.columns))
    for idx in common_indexes:
        rna_graph.loc[idx, idx] = 1

    # --- Step 2: MetaPath2Vec Enhancement ---
    rna_pca = perform_pca_scanpy(adata_rna, n_components=pca_components)
    atac_pca = perform_pca_scanpy(adata_atac, n_components=pca_components)

    print("Running MetaPath2Vec to refine embeddings...")
    gene_embeddings, peak_embeddings = integrate_data_with_metapath2vec(rna_pca, atac_pca, rna_graph)

    # --- Step 3: Graph Refinement ---
    X_ref_refined = gene_embeddings
    X_query_refined = peak_embeddings
    
    nn = NearestNeighbors(n_neighbors=refine_k, metric='euclidean')
    nn.fit(X_ref_refined)
    _, indices_rna = nn.kneighbors(X_query_refined)
    
    nn = NearestNeighbors(n_neighbors=refine_k, metric='euclidean')
    nn.fit(X_query_refined)
    _, indices_atac = nn.kneighbors(X_ref_refined)

    adj_matrix = pd.DataFrame(create_adjacency_matrix(indices_rna, indices_atac))
    adj_matrix.index = adata_rna.obs_names
    adj_matrix.columns = adata_atac.obs_names
    rna_graph = adj_matrix.copy()

    for idx in common_indexes:
        rna_graph.loc[idx, idx] = 1

    # --- Step 4: Format Outputs ---
    enh_rna_train = pd.DataFrame(initial_pca[0:adata_rna.n_obs].values, index=adata_rna.obs_names)
    enh_atac_train = pd.DataFrame(initial_pca[adata_rna.n_obs:].values, index=adata_atac.obs_names)

    rows, cols = np.where(rna_graph.values > 0)
    values = np.ones_like(rows)
    
    rna_graph_train2 = pd.DataFrame(np.column_stack((rows, cols + rna_graph.shape[0], values)))

    return adata_rna, adata_atac, enh_rna_train, enh_atac_train, rna_graph_train2
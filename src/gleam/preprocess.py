import pandas as pd
import numpy as np
import random
import torch
import scanpy as sc
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
from scipy.sparse import issparse

def generate_adjacency_matrix(embeddings, n_neighbors=10):
    """
    Generates a sparse, symmetric adjacency matrix representing the cell-cell similarity graph.
    
    Logic:
    1. Uses kNN on the input embeddings (e.g., PCA) to find the closest cells.
    2. Constructs a sparse matrix where A[i, j] = 1 if i and j are neighbors.
    3. Symmetrizes the matrix to ensure the graph is undirected (A = A + A.T).
    """
    n_samples = embeddings.shape[0]
    # Find k nearest neighbors in the embedding space
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    # Fill sparse matrix
    adjacency_matrix = sparse.lil_matrix((n_samples, n_samples))
    for i in range(n_samples):
        adjacency_matrix[i, indices[i]] = 1
        
    # Symmetrize to make edges undirected
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix.data = np.ones_like(adjacency_matrix.data) # Binarize weights to 1
    return adjacency_matrix.tolil()


def generate_imputed_matrix(embeddings, full_matrix, n_neighbors=10):
    """
    Smooths a feature matrix by averaging the values of nearest neighbors.
    Useful for creating a 'denoised' reference or checking technical zeros.
    """
    full_matrix = np.array(full_matrix.values)
    n_samples, n_features = full_matrix.shape
    
    # Identify neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
    nn.fit(embeddings)
    indices = nn.kneighbors(embeddings, return_distance=False)
    indices = indices[:, 1:] # Exclude self
    
    # Gather neighbor values
    neighbor_values = full_matrix[indices]
    mask = np.isnan(full_matrix)
    valid_neighbor_mask = ~np.isnan(neighbor_values)
    
    # Compute average of valid (non-NaN) neighbors
    neighbor_sums = np.nansum(neighbor_values, axis=1)
    neighbor_counts = np.sum(valid_neighbor_mask, axis=1)
    neighbor_counts = np.maximum(neighbor_counts, 1) # Avoid divide by zero
    imputed_values = neighbor_sums / neighbor_counts
    
    # Replace values based on mask logic (Note: original logic seems to keep NaNs where mask is True)
    imputed_matrix = np.where(mask, full_matrix, imputed_values)
    return imputed_matrix


def calculate_tech_zero_proportion(data, similarity, include_second_degree=True):
    """
    Estimates the 'dropout' probability for zero values.
    
    Logic:
    If a cell has 0 expression for a gene, but its neighbors express it highly,
    it is likely a 'technical zero' (dropout). If neighbors also have 0, it is 
    likely a 'biological zero'.
    
    Returns:
    - proportions: Matrix where value represents the fraction of neighbors expressing the gene.
    """
    data_bool = data.astype(bool) # Convert counts to binary (expressed vs not)
    first_degree = similarity.astype(bool)
    
    # Optionally expand neighborhood to 2-hops for robust estimation
    if include_second_degree:
        second_degree = (similarity @ similarity).astype(bool)
        second_degree = second_degree - first_degree # Remove direct neighbors
        second_degree = second_degree.tolil()
        second_degree.setdiag(0) # Remove self-loops
        neighbors = first_degree + second_degree
    else:
        neighbors = first_degree
        
    # Sum of boolean expression across neighbors
    neighbor_sums = neighbors.dot(data_bool.values)
    total_neighbors = neighbors.sum(axis=1).A1
    
    # Calculate proportion
    proportions = pd.DataFrame(neighbor_sums / total_neighbors[:, np.newaxis], 
                               index=data.index, columns=data.columns)
    return proportions


def load_from_DF(datalist, emb, similarity, zero_inflate):
    """
    Main data loader that converts Pandas DataFrames into PyG HeteroData.
    
    Parameters:
    - datalist: List of DataFrames [RNA, ATAC]
    - emb: List of embeddings matching datalist
    - similarity: Cell-cell adjacency matrix (or coordinate list format)
    - zero_inflate: List of flags [0, 1] indicating if modality has dropouts (1=Yes)
    """
    data = HeteroData()
    cell_len = 0
    feat_name = list()
    conn_name = list()
    
    # 1. Initialize Node Metadata
    for i in range(len(datalist)):
        cell_len += emb[i].shape[0]
        if datalist[i] is None:
            continue
        feat_name.append('feat' + str(i)) # e.g., feat0 (Genes), feat1 (Peaks)
        conn_name.append('conn' + str(i))
        data[feat_name[i]].node_id = torch.arange(datalist[i].shape[1])
        
    data["cell"].node_id = torch.arange(cell_len)
    data['cell'].x =  torch.tensor(pd.concat(emb).values) # Cell features = embeddings
    
    # 2. Build Cell-Cell Similarity Graph
    edges = []
    edge_weights = []
    # Assuming 'similarity' is a DF with columns [row, col, weight]
    for cell_idx in range(similarity.shape[0]):
        edges.append([similarity.iloc[cell_idx, 0], similarity.iloc[cell_idx, 1]])
        edge_weights.append(similarity.iloc[cell_idx, 2])
        
    edge_index = torch.tensor(edges).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    data['cell', 'similarity', 'cell'].edge_index = edge_index
    data['cell', 'similarity', 'cell'].edge_weight = edge_weight
    data = T.ToUndirected()(data) # Ensure undirected graph
    
    # 3. Build Cell-Feature Bipartite Graphs
    incre_cell_len = 0
    for i in range(len(datalist)):
        if datalist[i] is None:
            incre_cell_len += emb[i].shape[0]
            continue
            
        # Case A: No Zero Inflation (e.g., standard regression)
        if zero_inflate[i] == 0:
            # Create a dense graph: Every cell connected to every feature
            cell_indices, feat_indices = np.meshgrid(np.arange(datalist[i].shape[0]), np.arange(datalist[i].shape[1]), indexing='ij')
            edges = np.column_stack((cell_indices.ravel() + incre_cell_len, feat_indices.ravel()))
            edge_weights = datalist[i].values.flatten()
            
            # Construct PyG edges
            edges = edges.tolist()
            edge_weights = edge_weights.tolist()     
            edge_index = torch.tensor(edges).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            
            data['cell', conn_name[i], feat_name[i]].edge_index = edge_index
            data['cell', conn_name[i], feat_name[i]].edge_weight = edge_weight
            data['cell', conn_name[i], feat_name[i]].mode = "Train"
            data['cell', conn_name[i], feat_name[i]].zero_inflation = 0
            
            # Store full matrix for loss calculation
            data['cell', conn_name[i], feat_name[i]].matrix = torch.Tensor(datalist[i].values)
            
        # Case B: Zero Inflation (e.g., scRNA-seq)
        else:
            # Calculate dropout probability map
            within_mod_similarity = generate_adjacency_matrix(emb[i], n_neighbors = 15)
            nonzero_proportions = calculate_tech_zero_proportion(datalist[i], within_mod_similarity)
            prop_zeros = sum(np.sum(datalist[i] == 0, axis = 0)) / (datalist[i].shape[0] * datalist[i].shape[1])
            
            edges = []
            edge_weights = []
            edges_all = []
            edge_weights_all = []
            edge_label = []
            
            data_numpy = datalist[i].to_numpy()
            nonzero_props = nonzero_proportions.to_numpy()
            
            # Mask creation for training
            # 1. Keep all non-zero values (Positive samples)
            mask_positive = (nonzero_props > 0) & (data_numpy > 0)
            
            # 2. Keep 'Sparse Zeros' (High likelihood of being true biological zeros)
            # Logic: If data is 0 AND neighbors mostly have 0 (prop <= 0.8), assume true zero.
            mask_sparse_zero = (nonzero_props <= 0.8) & (data_numpy == 0)

            # Build Positive Edges
            cell_indices, feat_indices = np.meshgrid(np.arange(data_numpy.shape[0]), np.arange(data_numpy.shape[1]), indexing='ij')
            positive_edges = np.column_stack((cell_indices[mask_positive] + incre_cell_len, feat_indices[mask_positive]))
            positive_weights = data_numpy[mask_positive]
            
            edges.extend(positive_edges.tolist())
            edge_weights.extend(positive_weights.tolist())
            edge_label.extend([1] * len(positive_weights)) # Label 1 = Real Expression
            
            edge_weights_all.extend(positive_weights.tolist())
            edges_all.extend(positive_edges.tolist())
            
            # Build Negative Edges (Biological Zeros)
            # Randomly sample from the sparse zeros based on global zero proportion to balance data
            sparse_zero_mask = (np.random.random(data_numpy.shape) > prop_zeros)
            sparse_zero_edges = np.column_stack((cell_indices[sparse_zero_mask] + incre_cell_len, feat_indices[sparse_zero_mask]))
            
            edge_label.extend([0] * sparse_zero_mask.sum()) # Label 0 = True Zero
            edge_weights_all.extend([0] * sparse_zero_mask.sum())
            edges_all.extend(sparse_zero_edges.tolist())      
            
            # Construct tensors
            edge_index = torch.tensor(edges).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            data['cell', conn_name[i], feat_name[i]].edge_index = edge_index
            data['cell', conn_name[i], feat_name[i]].edge_weight = edge_weight
            
            # 'edge_label_index' contains both Pos and Neg samples for classification loss
            edge_index_all = torch.tensor(edges_all).t().contiguous()
            edge_weights_all = torch.tensor(edge_weights_all, dtype=torch.float)
            edge_label = torch.tensor(edge_label, dtype=torch.float)
            
            data['cell', conn_name[i], feat_name[i]].edge_label = edge_label
            data['cell', conn_name[i], feat_name[i]].edge_label_index = edge_index_all
            data['cell', conn_name[i], feat_name[i]].edge_label_weight = edge_weights_all
            data["cell", conn_name[i], feat_name[i]].mode = "Train"
            data['cell', conn_name[i], feat_name[i]].zero_inflation = 1
            
            # Binary mask for classification auxiliary tasks
            binary_data = datalist[i].copy(deep=True)
            binary_data[binary_data > 0] = 1
            data['cell', conn_name[i], feat_name[i]].matrix = torch.Tensor(binary_data.values)
            
        incre_cell_len += datalist[i].shape[0]
    return(data)


def pred_from_DF(emb, feat, zero_inflate, conn_name, feat_name, knn_matrix = None):
    """
    Constructs a HeteroData object for inference (Prediction phase).
    Unlike training, this creates a dense graph where every query cell is connected
    to every target feature, allowing the model to predict a full profile.
    """
    data = HeteroData()
    data[feat_name].node_id = torch.arange(feat.shape[1])
    data["cell"].node_id = torch.arange(emb.shape[0])
    data['cell'].x = torch.tensor(emb)
    
    # Create dense edges (All cells x All features)
    edges = []
    edge_weights = []
    cell_indices, feat_indices = np.meshgrid(np.arange(emb.shape[0]), np.arange(feat.shape[1]), indexing='ij')
    edges = np.column_stack((cell_indices.ravel(), feat_indices.ravel()))
    
    # Optional: Use kNN weights if provided (usually None for pure prediction)
    if knn_matrix is not None:
        edge_weights = knn_matrix.values.flatten()
        edge_weights = edge_weights.tolist()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        data['cell', conn_name, feat_name].edge_weight = edge_weight
    else:
        edge_weights = None
        
    edges = edges.tolist()
    edge_index = torch.tensor(edges).t().contiguous()
    
    data['cell', conn_name, feat_name].edge_index = edge_index
    data['cell', conn_name, feat_name].mode = "Pred"
    
    # Set flags for the model's forward pass
    if zero_inflate == 0:
        data['cell', conn_name, feat_name].zero_inflation = 0
    else:
        data['cell', conn_name, feat_name].zero_inflation = 1
        data['cell', conn_name, feat_name].edge_label_index = edge_index # Evaluate on all edges
        
    data = T.ToUndirected()(data)
    data['cell'].x = data['cell'].x.to(torch.float32)
    return(data)


def find_kNN(emb, k = 10):
    """
    Finds k-Nearest Neighbors across different datasets (batches).
    Useful for aligning multiple datasets or checking integration quality.
    """
    results = []
    # Calculate index offsets to handle global indexing
    index_offsets = np.cumsum([0] + [df.shape[0] for df in emb[:-1]])
    
    for i, df in enumerate(emb):
        # Combine all other datasets to serve as the reference for neighbors
        other_dfs = [emb[j] for j in range(len(emb)) if j != i]
        combined_df = pd.concat(other_dfs, ignore_index=True)
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(combined_df)
        distances, indices = nbrs.kneighbors(df)
        
        # Normalize distances to 0-1 range
        scaler = MinMaxScaler()
        normalized_distances = scaler.fit_transform(distances)
        
        # Map local indices back to global indices
        for idx in range(len(df)):
            global_idx = idx + index_offsets[i]
            for neighbor_idx, distance in zip(indices[idx], normalized_distances[idx]):
                if neighbor_idx < index_offsets[i]:
                    neighbor_global_idx = neighbor_idx
                else:
                    neighbor_global_idx = neighbor_idx + df.shape[0]
                negative_distance = 1 - distance
                results.append((global_idx, neighbor_global_idx, negative_distance))
                
    results_df = pd.DataFrame(results, columns=['data_i', 'data_j', 'negative_distance'])
    return results_df


def integrate_rna_atac_scanpy(rna, atac, n_components=50):
    """
    Integrates RNA and ATAC data using Scanpy with zero-preserved normalization.
    
    Parameters:
    -----------
    rna : pd.DataFrame
        RNA expression data (rows=cells, cols=genes).
    atac : pd.DataFrame
        ATAC accessibility data (rows=cells, cols=genes/peaks).
        Note: Columns must overlap with RNA for integration (e.g., Gene Activity Scores).
    n_components : int
        Number of PCA components to generate.
        
    Returns:
    --------
    pd.DataFrame
        Integrated PCA embeddings corresponding to the concatenated structure:
        [RNA_cells, ATAC_cells] similar to the '_GLEAM_emb.csv' file.
    """
    
    # 1. Initialize AnnData objects
    adata_rna = sc.AnnData(rna)
    adata_atac = sc.AnnData(atac)
    
    # 2. Preprocessing: Find common features
    # Integration requires a shared feature space (intersection of genes)
    common_features = np.intersect1d(adata_rna.var_names, adata_atac.var_names)
    
    if len(common_features) == 0:
        raise ValueError("No common features found between RNA and ATAC. "
                         "Ensure ATAC columns are Gene Activity Scores matching RNA gene symbols.")
    
    print(f"Integrating based on {len(common_features)} common features...")
    
    # Subset to common features
    adata_rna = adata_rna[:, common_features].copy()
    adata_atac = adata_atac[:, common_features].copy()
    
    # 3. Concatenate along the cell axis (RNA on top, ATAC on bottom)
    # This matches the structure of your X_ref (RNA) and X_query (ATAC) slicing
    adata_joint = sc.concat([adata_rna, adata_atac], keys=['RNA', 'ATAC'], index_unique=None)
    
    # 4. Standard Normalize & Log Transform (if not already done)
    # Checks if data looks raw (max value > 20 is a common heuristic)
    if np.max(adata_joint.X) > 20:
        print("Detected raw counts. Normalizing and log-transforming...")
        sc.pp.normalize_total(adata_joint, target_sum=1e4)
        sc.pp.log1p(adata_joint)
        
    # 5. Zero-Preserved Normalization
    # 'zero_center=False' ensures we scale variance without shifting dense zeros to non-zeros
    print("Applying zero-preserved scaling...")
    sc.pp.scale(adata_joint, max_value=10, zero_center=False)
    
    # 6. Run PCA
    # Note: sc.tl.pca uses the scaled data in .X
    print(f"Running PCA with {n_components} components...")
    sc.tl.pca(adata_joint, n_comps=n_components)
    
    # 7. Extract Embeddings
    pca_df = pd.DataFrame(
        adata_joint.obsm['X_pca'],
        index=adata_joint.obs_names
    )
    
    return pca_df


def load_from_AnnData(datalist, emb, similarity, zero_inflate):
    """
    Main data loader that converts a list of Scanpy AnnData objects into PyG HeteroData.
    
    Parameters:
    - datalist: List of AnnData objects [RNA, ATAC]
    - emb: List of embeddings (DataFrames) matching datalist
    - similarity: Cell-cell adjacency matrix (or coordinate list format)
    - zero_inflate: List of flags [0, 1] indicating if modality has dropouts (1=Yes)
    """
    data = HeteroData()
    cell_len = 0
    feat_name = list()
    conn_name = list()
    
    # 1. Initialize Node Metadata
    for i in range(len(datalist)):
        cell_len += emb[i].shape[0]
        if datalist[i] is None:
            continue
        feat_name.append('feat' + str(i))
        conn_name.append('conn' + str(i))
        # Use .n_vars from AnnData
        data[feat_name[i]].node_id = torch.arange(datalist[i].n_vars)
        
    data["cell"].node_id = torch.arange(cell_len)
    data['cell'].x =  torch.tensor(pd.concat(emb).values)
    
    # 2. Build Cell-Cell Similarity Graph
    edges = []
    edge_weights = []
    for cell_idx in range(similarity.shape[0]):
        edges.append([similarity.iloc[cell_idx, 0], similarity.iloc[cell_idx, 1]])
        edge_weights.append(similarity.iloc[cell_idx, 2])
        
    edge_index = torch.tensor(edges).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    data['cell', 'similarity', 'cell'].edge_index = edge_index
    data['cell', 'similarity', 'cell'].edge_weight = edge_weight
    data = T.ToUndirected()(data)
    
    # 3. Build Cell-Feature Bipartite Graphs
    incre_cell_len = 0
    for i in range(len(datalist)):
        adata = datalist[i]
        if adata is None:
            incre_cell_len += emb[i].shape[0]
            continue
            
        # Extract data matrix (dense numpy array)
        if issparse(adata.X):
            data_numpy = adata.X.toarray()
        else:
            data_numpy = adata.X
            
        # Case A: No Zero Inflation
        if zero_inflate[i] == 0:
            cell_indices, feat_indices = np.meshgrid(np.arange(adata.n_obs), np.arange(adata.n_vars), indexing='ij')
            edges = np.column_stack((cell_indices.ravel() + incre_cell_len, feat_indices.ravel()))
            edge_weights = data_numpy.flatten()
            
            edges = edges.tolist()
            edge_weights = edge_weights.tolist()     
            edge_index = torch.tensor(edges).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            
            data['cell', conn_name[i], feat_name[i]].edge_index = edge_index
            data['cell', conn_name[i], feat_name[i]].edge_weight = edge_weight
            data['cell', conn_name[i], feat_name[i]].mode = "Train"
            data['cell', conn_name[i], feat_name[i]].zero_inflation = 0
            data['cell', conn_name[i], feat_name[i]].matrix = torch.Tensor(data_numpy)
            
        # Case B: Zero Inflation
        else:
            # Note: You might need to implement a sparse version of generate_adjacency_matrix if data is huge
            # For now, assuming embeddings are dense
            within_mod_similarity = generate_adjacency_matrix(emb[i], n_neighbors = 15)
            
            # Helper to calculate proportions using the dense data extracted from AnnData
            # We temporarily convert to DF for the helper function (or refactor helper to take arrays)
            temp_df = pd.DataFrame(data_numpy, index=adata.obs_names, columns=adata.var_names)
            nonzero_proportions = calculate_tech_zero_proportion(temp_df, within_mod_similarity)
            
            prop_zeros = np.sum(data_numpy == 0) / (adata.n_obs * adata.n_vars)
            
            edges = []
            edge_weights = []
            edges_all = []
            edge_weights_all = []
            edge_label = []
            
            nonzero_props = nonzero_proportions.to_numpy()
            mask_positive = (nonzero_props > 0) & (data_numpy > 0)
            mask_sparse_zero = (nonzero_props <= 0.8) & (data_numpy == 0)

            cell_indices, feat_indices = np.meshgrid(np.arange(adata.n_obs), np.arange(adata.n_vars), indexing='ij')
            
            # Positives
            positive_edges = np.column_stack((cell_indices[mask_positive] + incre_cell_len, feat_indices[mask_positive]))
            positive_weights = data_numpy[mask_positive]
            
            edges.extend(positive_edges.tolist())
            edge_weights.extend(positive_weights.tolist())
            edge_label.extend([1] * len(positive_weights))
            
            edge_weights_all.extend(positive_weights.tolist())
            edges_all.extend(positive_edges.tolist())
            
            # Negatives (Sparse Zeros)
            sparse_zero_mask = (np.random.random(data_numpy.shape) > prop_zeros)
            sparse_zero_edges = np.column_stack((cell_indices[sparse_zero_mask] + incre_cell_len, feat_indices[sparse_zero_mask]))
            
            edge_label.extend([0] * sparse_zero_mask.sum())
            edge_weights_all.extend([0] * sparse_zero_mask.sum())
            edges_all.extend(sparse_zero_edges.tolist())      
            
            edge_index = torch.tensor(edges).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            data['cell', conn_name[i], feat_name[i]].edge_index = edge_index
            data['cell', conn_name[i], feat_name[i]].edge_weight = edge_weight
            
            edge_index_all = torch.tensor(edges_all).t().contiguous()
            edge_weights_all = torch.tensor(edge_weights_all, dtype=torch.float)
            edge_label = torch.tensor(edge_label, dtype=torch.float)
            
            data['cell', conn_name[i], feat_name[i]].edge_label = edge_label
            data['cell', conn_name[i], feat_name[i]].edge_label_index = edge_index_all
            data['cell', conn_name[i], feat_name[i]].edge_label_weight = edge_weights_all
            data["cell", conn_name[i], feat_name[i]].mode = "Train"
            data['cell', conn_name[i], feat_name[i]].zero_inflation = 1
            
            binary_data = data_numpy.copy()
            binary_data[binary_data > 0] = 1
            data['cell', conn_name[i], feat_name[i]].matrix = torch.Tensor(binary_data)
            
        incre_cell_len += adata.n_obs
    return(data)


def pred_from_AnnData(emb, adata_feat, zero_inflate, conn_name, feat_name, knn_matrix = None):
    """
    Constructs HeteroData for prediction using an AnnData object as target features.
    """
    data = HeteroData()
    data[feat_name].node_id = torch.arange(adata_feat.n_vars)
    data["cell"].node_id = torch.arange(emb.shape[0])
    data['cell'].x = torch.tensor(emb)
    
    edges = []
    edge_weights = []
    # Using n_obs from embedding (query cells) and n_vars from adata (target features)
    cell_indices, feat_indices = np.meshgrid(np.arange(emb.shape[0]), np.arange(adata_feat.n_vars), indexing='ij')
    edges = np.column_stack((cell_indices.ravel(), feat_indices.ravel()))
    
    if knn_matrix is not None:
        edge_weights = knn_matrix.values.flatten()
        edge_weights = edge_weights.tolist()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        data['cell', conn_name, feat_name].edge_weight = edge_weight
    else:
        edge_weights = None
        
    edges = edges.tolist()
    edge_index = torch.tensor(edges).t().contiguous()
    
    data['cell', conn_name, feat_name].edge_index = edge_index
    data['cell', conn_name, feat_name].mode = "Pred"
    if zero_inflate == 0:
        data['cell', conn_name, feat_name].zero_inflation = 0
    else:
        data['cell', conn_name, feat_name].zero_inflation = 1
        data['cell', conn_name, feat_name].edge_label_index = edge_index
        
    data = T.ToUndirected()(data)
    data['cell'].x = data['cell'].x.to(torch.float32)
    return(data)
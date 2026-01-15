import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, GraphConv
from torch_geometric.nn.conv import MessagePassing
from typing import List, Optional, Tuple, Union
from torch_sparse import SparseTensor, spmm
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch.nn import Linear
from torch_geometric.nn.aggr import Aggregation
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import os


class CustomGraphConv(MessagePassing):
    """
    A custom Graph Convolution layer that extends PyG's MessagePassing.
    
    Key Features:
    - Differentiates between 'same-type' edges (e.g., cell-cell similarity) and 
      'cross-type' edges (e.g., cell-gene).
    - Applies specific scalar weights (`same_type_weight`, `cross_type_weight`) 
      during message passing to balance the influence of homophily vs. heterophily.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        node_id_dict: dict,
        same_type_weight: float = 0.3,
        cross_type_weight: float = 1.0, 
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.same_type_weight = same_type_weight
        self.cross_type_weight = cross_type_weight
        self.normalize = normalize
        self.root_weight = root_weight
        self.node_id_dict = node_id_dict # Dictionary containing ID ranges for node types

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # Base convolution layer
        self.conv = GraphConv(in_channels[0], out_channels, aggr=aggr, bias=bias)

        # Linear layer for the root node (self-loop transformation)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        size: Size = None,
    ) -> Tensor:
        """"
        Standard forward pass for message passing.
        """
        if isinstance(x, Tensor):
            x = (x, x)

        # Propagate messages along edges
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        # Add root node features (skip connection/self-loop logic)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        # Optional L2 normalization
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_index: Adj, edge_weight: Optional[Tensor]) -> Tensor:
        """
        Constructs messages. This is where the custom weighting logic happens.
        """
        row, col = edge_index
        cell_len = len(self.node_id_dict['cell'])
        
        # Determine if nodes are cells (type 0) or features (type 1) based on index
        src_types = self.get_node_types(row, cell_len)
        tgt_types = self.get_node_types(col, cell_len)

        # Apply specific weights based on whether source and target are the same type
        type_weight = torch.where(
            src_types == tgt_types,
            torch.full_like(src_types, self.same_type_weight, dtype=torch.float),  # e.g., cell-cell
            torch.full_like(src_types, self.cross_type_weight, dtype=torch.float)  # e.g., cell-gene
        )

        # Combine learned edge weights with the structural type weight
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1, 1)
        else:
            edge_weight = torch.ones_like(type_weight, dtype=torch.float).view(-1, 1)
            
        weight = type_weight.view(-1, 1) * edge_weight
        
        # Message = neighbor_feature * combined_weight
        out = x_j * weight
        return out

    def get_node_types(self, indices, cell_len):
        """Helper to classify nodes based on index thresholds."""
        types = torch.full_like(indices, -1, dtype=torch.long)
        types[indices <= cell_len] = 0  # Is Cell
        types[indices > cell_len] = 1   # Is Feature (Gene/Peak)
        return types

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


class Encoder(torch.nn.Module):
    """
    Graph Neural Network Encoder.
    Consists of two stacked CustomGraphConv layers.
    """
    def __init__(self, hidden_channels, node_id_dict):
        super().__init__()
        self.conv1 = CustomGraphConv(hidden_channels, hidden_channels, node_id_dict, aggr = "mean")
        self.conv2 = CustomGraphConv(hidden_channels, hidden_channels, node_id_dict, aggr = "mean")

    def forward(self, x, edge_index, edge_weight):
        # Two-hop message passing
        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class EdgePredictor(torch.nn.Module):
    """
    Binary Classifier (MLP) to predict the existence of an edge (Drop-out vs. True Zero).
    Output dimension is 1 (Logit).
    """
    def __init__(self, input_dim, hidden_dim):
        super(EdgePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = torch.nn.Linear(hidden_dim // 2, 1)

    def forward(self, combined_feat):
        x = F.leaky_relu(self.fc1(combined_feat))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class WeightPredictor(torch.nn.Module):
    """
    Regressor (MLP) to predict the weight/expression value of an edge.
    """
    def __init__(self, input_dim, hidden_dim):
        super(WeightPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = torch.nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, combined_feat):
        x = F.leaky_relu((self.fc1(combined_feat)))
        x = F.leaky_relu((self.fc2(x)))
        x = F.leaky_relu((self.fc3(x)))
        x = self.fc4(x)
        return x

    
class WeightPredictorTransformer(nn.Module):
    """
    Transformer-based Decoder.
    Takes cell embeddings and predicts the entire feature vector for that cell.
    Useful for global reconstruction of gene expression/chromatin accessibility.
    """
    def __init__(self, input_dim, out_dim, hidden_dim=512, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        # Project input embedding to transformer dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection to feature space size (out_dim = num_genes or num_peaks)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_embeddings):
        x = self.norm1(self.input_proj(cell_embeddings))
        x = x.unsqueeze(1) # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)   
        x = self.norm2(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Project to final output dimension
        return x
    

    
    
class RegressorHybrid(torch.nn.Module):
    """
    Hybrid Decoder that combines:
    1. Edge-based MLPs (EdgePredictor/WeightPredictor) for link prediction.
    2. Transformer-based reconstruction for global profile prediction.
    """
    def __init__(self, hidden_channels, edge_types, feat_dim=None):
        super().__init__()
        # Original edge-based predictors for each specific edge type (e.g. cell-gene)
        self.edge_exist = torch.nn.ModuleDict({
            str(edge_type): EdgePredictor(2 * hidden_channels, hidden_channels) 
            for edge_type in edge_types
        })
        self.edge_weight = torch.nn.ModuleDict({
            str(edge_type): WeightPredictor(2 * hidden_channels, hidden_channels)
            for edge_type in edge_types
        })
        
        # Transformer predictor for creating full profiles
        if feat_dim is not None:
            self.transformer_predictor = torch.nn.ModuleDict({
                str(edge_types[i]): WeightPredictorTransformer(hidden_channels, feat_dim[i])
                for i in range(len(edge_types))
            })

    def forward(self, x_src: Tensor, x_dst: Tensor, edge_type, edge_label_index, mode, 
                edge_label=None, zero_inflation=None, use_transformer=False):
        if use_transformer:
            # Predict full profile for unique source cells
            # print(x_src[edge_label_index[0].unique()].shape)
            return self.transformer_predictor[str(edge_type)](x_src[edge_label_index[0].unique()])
        else:
            # Edge-based prediction (Link Prediction)
            edge_feat_src = x_src[edge_label_index[0]]
            edge_feat_dst = x_dst[edge_label_index[1]]
            # Concatenate source and target embeddings
            combined_feat = torch.cat([edge_feat_src, edge_feat_dst], dim=-1)
            
            if zero_inflation == 1:
                # Predict existence (Probability of non-dropout)
                link_exists = self.edge_exist[str(edge_type)](combined_feat)
                
                # Predict weight (Expression Level)
                if mode != "Pred":
                    # During training, only predict weights for non-zero edges to compute MSE
                    link_weight = self.edge_weight[str(edge_type)](combined_feat[edge_label == 1])
                else:
                    # During prediction, predict for all
                    link_weight = self.edge_weight[str(edge_type)](combined_feat)
                return link_exists, link_weight
            else:
                # Simple regression for non-zero-inflated data
                link_weight = self.edge_weight[str(edge_type)](combined_feat)
                return link_weight
            

            
class Model(torch.nn.Module):
    """
    Main GLEAM Model.
    Integrates the Input Projection, HeteroGNN Encoder, and Hybrid Decoder.
    """
    def __init__(self, data, input_channels, hidden_channels):
        super().__init__()
        # Project cell input features (PCA) to hidden dimension
        self.cell_shared_emb = torch.nn.Linear(input_channels, hidden_channels)
        
        # Learnable embeddings for features (Genes/Peaks) which act as nodes
        self.embeddings = torch.nn.ModuleDict()
        feat_dims = []
        for feature in data.node_types:
            if feature.startswith('feat'):
                num_nodes = data[feature].num_nodes
                feat_dims.append(num_nodes)
                self.embeddings[feature] = torch.nn.Embedding(num_nodes, hidden_channels)
        
        # Initialize GNN and convert to HeteroGNN
        self.gnn = Encoder(hidden_channels, data.node_id_dict)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        
        # Identify relevant edge types for regression (excluding cell-cell similarity)
        all_edge_types = []
        for edge_type in data.edge_types:
            src, relation, dst = edge_type
            if dst == 'cell':
                continue
            all_edge_types.append(f"{src}-{relation}-{dst}")
        
        # Initialize Decoder
        self.regressor = RegressorHybrid(hidden_channels, all_edge_types, feat_dim = feat_dims)
        self.projection = torch.nn.Linear(hidden_channels, input_channels)
        self.cache = {}

    def forward(self, data: HeteroData) -> Tensor:
        # 1. Prepare Initial Node Embeddings
        x_dict = {
            "cell": self.cell_shared_emb(data["cell"].x.float())
        }
        for feature in self.embeddings.keys():
            x_dict[feature] = self.embeddings[feature](data[feature].node_id)
        
        # 2. Run GNN Encoder
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_weight_dict)
        self.cache['saved_embeddings'] = x_dict
        
        preds = {}    
        preds['transformer'] = {}

        # 3. Decode / Predict
        for edge_type in data.edge_types:
            src, relation, dst = edge_type
            if dst == 'cell':
                continue
                
            mode = data[edge_type].mode
            zero_inflation = data[edge_type].zero_inflation
            
            # Use label indices for training (masked) or full indices for prediction
            if zero_inflation == 1:
                edge_label_index = data[edge_type].edge_label_index
                edge_label=data[edge_type].edge_label
            else:
                edge_label_index = data[edge_type].edge_index
                edge_label = None
                
            pred_key = f"{src}-{relation}-{dst}"
            
            # A. Transformer Prediction (Global Profile)
            preds['transformer'][pred_key] = self.regressor(
                    x_src=x_dict[src],
                    x_dst=x_dict[dst],
                    edge_type=pred_key,
                    edge_label_index=edge_label_index,
                    mode=mode,
                    use_transformer=True
                )
            
            # B. Edge-based Prediction (Chunked to save memory)
            if edge_label_index.shape[1] < 300000:
                preds[pred_key] = self.regressor(
                    x_src=x_dict[src],
                    x_dst=x_dict[dst],
                    edge_type=pred_key,
                    edge_label_index=edge_label_index,
                    mode=mode,
                    edge_label=edge_label,
                    zero_inflation=zero_inflation,
                    use_transformer=False
                )
            else:
                # Process large graphs in chunks of 300,000 edges
                if zero_inflation == 0:
                    out = torch.tensor([])
                    if (edge_label_index.get_device() >= 0):
                        out = out.to(edge_label_index.get_device())
                    for i in range(edge_label_index.shape[1] // 300000 + 1):
                        preded = self.regressor(
                            x_src=x_dict[src],
                            x_dst=x_dict[dst],
                            edge_type=pred_key,
                            edge_label_index = edge_label_index[:, (300000 * i): (300000 * (i + 1))],
                            mode=mode,
                            zero_inflation=zero_inflation,
                            use_transformer=False
                        )
                        out = torch.cat((out, preded), 0)
                    preds[pred_key] = out
                else:
                    out0 = torch.tensor([])
                    out1 = torch.tensor([])
                    if (edge_label_index.get_device() >= 0):
                        out0 = out0.to(edge_label_index.get_device())
                        out1 = out1.to(edge_label_index.get_device())
                    for i in range(edge_label_index.shape[1] // 300000 + 1):
                        preded = self.regressor(
                            x_src=x_dict[src],
                            x_dst=x_dict[dst],
                            edge_type=pred_key,
                            edge_label_index = edge_label_index[:, (300000 * i): (300000 * (i + 1))],
                            mode=mode,
                            edge_label=edge_label[(300000 * i): (300000 * (i + 1))],
                            zero_inflation=zero_inflation,
                            use_transformer=False
                        )
                        out0 = torch.cat((out0, preded[0]), 0)
                        out1 = torch.cat((out1, preded[1]), 0)
                    preds[pred_key] = [out0, out1]
        preds['cell'] = x_dict['cell']
        return preds
        
    def get_node_embeddings(self, data: HeteroData):
        """Helper to run just the encoder part."""
        x_dict = {
            "cell": self.cell_shared_emb(data["cell"].x.float())
        }
        for feature in self.embeddings.keys():
            x_dict[feature] = self.embeddings[feature](data[feature].node_id)
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_weight_dict)
        return x_dict

    def predict(self, data: HeteroData):
        """Prediction mode using cached embeddings."""
        x_dict = self.cache['saved_embeddings']
        preds = {}
        preds['transformer'] = {}
        
        for edge_type in data.edge_types:
            src, relation, dst = edge_type
            if dst == 'cell':
                continue
            edge_label_index = data[edge_type].edge_index
            mode = data[edge_type].mode
            zero_inflation = data[edge_type].zero_inflation
            pred_key = f"{src}-{relation}-{dst}"
            upper_shape = 100000 # Smaller chunk size for inference safety
            
            # Transformer Prediction
            preds['transformer'][pred_key] = self.regressor(
                x_src=data['cell'].x,
                x_dst=x_dict[dst],
                edge_type=pred_key,
                edge_label_index=edge_label_index,
                mode=mode,
                use_transformer=True
            )
            
            # Edge Prediction with Chunking
            if edge_label_index.shape[1] < upper_shape:
                preds[pred_key] = self.regressor(
                    x_src=data['cell'].x,
                    x_dst=x_dict[dst],
                    edge_type=pred_key,
                    edge_label_index=edge_label_index,
                    mode=mode,
                    zero_inflation=zero_inflation,
                    use_transformer=False
                )
            else:
                if zero_inflation == 0:
                    out = torch.tensor([])
                    if (edge_label_index.get_device() >= 0):
                        out = out.to(edge_label_index.get_device())
                    for i in range(edge_label_index.shape[1] // upper_shape + 1):
                        preded = self.regressor(
                            x_src=data['cell'].x,
                            x_dst=x_dict[dst],
                            edge_type=pred_key,
                            edge_label_index = edge_label_index[:, (upper_shape * i): (upper_shape * (i + 1))],
                            mode=mode,
                            zero_inflation=zero_inflation,
                            use_transformer=False
                        )
                        out = torch.cat((out, preded), 0)
                    preds[pred_key] = out
                else:
                    out0 = torch.tensor([])
                    out1 = torch.tensor([])
                    if (edge_label_index.get_device() >= 0):
                        out0 = out0.to(edge_label_index.get_device())
                        out1 = out1.to(edge_label_index.get_device())
                    for i in range(edge_label_index.shape[1] // upper_shape + 1):
                        preded = self.regressor(
                            x_src=data['cell'].x,
                            x_dst=x_dict[dst],
                            edge_type=pred_key,
                            edge_label_index = edge_label_index[:, (upper_shape * i): (upper_shape * (i + 1))],
                            mode=mode,
                            zero_inflation=zero_inflation,
                            use_transformer=False
                        )
                        out0 = torch.cat((out0, preded[0]), 0)
                        out1 = torch.cat((out1, preded[1]), 0)
                    preds[pred_key] = [out0, out1]
        return preds


class Trainer:
    """
    Handles the training loop with mixed precision support.
    """
    def __init__(self, model, data, input_channels, hidden_channels=150, lr=0.001, epochs=500):
        self.model = model(data=data, input_channels=input_channels, hidden_channels=hidden_channels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.data = data
        self.epochs = epochs
        self.scaler = GradScaler() # For Mixed Precision Training

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Automatic Mixed Precision
        with autocast():
            pred_result = self.model(self.data)
            loss = 0            
            for edge_type in self.data.edge_types:
                src, relation, dst = edge_type
                if dst == 'cell':
                    continue
                zero_inflation = self.data[edge_type].zero_inflation
                pred_key = f"{src}-{relation}-{dst}" 
                
                # Zero-Inflated Loss Calculation
                if zero_inflation == 1:
                    # 1. Existence Loss (BCE)
                    pred_binary = pred_result[pred_key][0].reshape(-1)
                    target_binary = self.data[(src, relation, dst)].edge_label
                    loss += F.binary_cross_entropy_with_logits(pred_binary, target_binary)
                    # 2. Weight Loss (MSE) - only on non-zero targets
                    pred = pred_result[pred_key][1].reshape(-1)
                    target = self.data[(src, relation, dst)].edge_weight
                    loss += F.mse_loss(pred, target)
                    
                    # 3. Transformer Loss (Weighted BCE for global profile)
                    if 'transformer' in pred_result:
                        transformer_pred = pred_result['transformer'][pred_key]
                        true_expression = self.data[(src, relation, dst)].matrix
                        # Weighting positive vs negative samples based on sparsity
                        density = true_expression.reshape(-1).mean()
                        weight_pos = 1.0 / (density + 1e-6)
                        weight_neg = 1.0 / (1.0 - density + 1e-6)
                        weights = true_expression.reshape(-1) * weight_pos + (1-true_expression.reshape(-1)) * weight_neg
                        transformer_loss = F.binary_cross_entropy_with_logits(transformer_pred.reshape(-1), true_expression.reshape(-1), weight=weights)
                        print(f"Transformer Loss: {transformer_loss:.4f}")
                        loss += transformer_loss
                else:
                    # Standard MSE Loss
                    pred = pred_result[pred_key].reshape(-1)
                    target = self.data[(src, relation, dst)].edge_weight
                    loss += F.mse_loss(pred, target)
                    if 'transformer' in pred_result:
                        transformer_pred = pred_result['transformer'][pred_key]
                        true_expression = self.data[(src, relation, dst)].matrix
                        transformer_loss = F.mse_loss(transformer_pred.reshape(-1), true_expression.reshape(-1))
                        loss += transformer_loss

            # 4. Embedding Regularization Loss (Keep embeddings close to input projection)
            emb_loss = torch.linalg.matrix_norm((pred_result['cell'] - self.model.cell_shared_emb(self.data['cell'].x.half())), ord=float('2'))
            loss = loss + 0.05 * emb_loss

        # Scale loss and backpropagate
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss)

    def train(self):
        best_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            loss = self.train_step()
            print(f'Epoch: {epoch:03d}, Total Loss: {loss:.4f}')
            
 
            
def predict(model, data):
    """Simple wrapper function for model prediction."""
    model.eval()
    with torch.no_grad():
        pred = model.predict(data)
    return(pred)


def predict_and_save(model, rna_train_shape, atac_train_shape, 
                     target_rna, target_atac, 
                     obs_rna, obs_atac, 
                     zero1, zero2, 
                     dataset_name, mod1_name="RNA", mod2_name="ATAC", 
                     output_dir="/storage10/siqishen/GLEAM/Pred/"):
    """
    Predicts cross-modality profiles using learned embeddings and saves the results to CSV.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained GLEAM model (on CPU).
    rna_train_shape : int
        Number of RNA cells in the training set (used for slicing embeddings).
    atac_train_shape : int
        Number of ATAC cells in the training set.
    target_rna : pd.DataFrame
        The RNA feature space (genes) to predict.
    target_atac : pd.DataFrame
        The ATAC feature space (peaks) to predict.
    obs_rna : pd.DataFrame
        The real observed RNA data (for saving as ground truth).
    obs_atac : pd.DataFrame
        The real observed ATAC data (for saving as ground truth).
    zero1 : int
        Zero-inflation flag for RNA (mod1).
    zero2 : int
        Zero-inflation flag for ATAC (mod2).
    dataset_name : str
        Name of the dataset (e.g., "Xu2022") for file naming.
    mod1_name : str
        Name of the first modality (default "RNA").
    mod2_name : str
        Name of the second modality (default "ATAC").
    output_dir : str
        Directory to save the output CSV files.
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cell_embeddings = model.cache['saved_embeddings']['cell']
    
    # --- Prediction 1: Predict ATAC (mod2) using RNA cell embeddings ---
    print(f"Predicting {mod2_name} from {mod1_name} embeddings...")
    
    # Slice embeddings for RNA cells (0 to N_rna)
    rna_cell_emb = cell_embeddings[0:rna_train_shape]
    
    # Prepare prediction data object for ATAC features ('conn1', 'feat1')
    pred_data_atac = pred_from_DF(rna_cell_emb, target_atac, zero2, 'conn1', 'feat1')
    pred_data_atac['cell'].x = pred_data_atac['cell'].x.to(torch.float32)
    pred_data_atac = pred_data_atac.to("cpu")
    
    # Run prediction
    pred_atac = predict(model, pred_data_atac)

    # Process and save ATAC predictions
    if zero2 == 1:
        # Extract weight (value) and existence (probability)
        all_pred = pred_atac['cell-conn1-feat1'][1].reshape(-1)
        all_exist = pred_atac['cell-conn1-feat1'][0].reshape(-1)
        
        # Save Weights
        d = pd.DataFrame({
            'row': pred_data_atac['cell', 'conn1', 'feat1'].edge_index[0].tolist(), 
            'col': pred_data_atac['cell', 'conn1', 'feat1'].edge_index[1].tolist(),
            'val': all_pred.cpu().detach().numpy()
        })
        matrix_df = d.pivot(index='row', columns='col', values='val')
        matrix_df.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod2_name}-weight.csv")
        
        # Save Existence Probabilities
        d = pd.DataFrame({
            'row': pred_data_atac['cell', 'conn1', 'feat1'].edge_index[0].tolist(), 
            'col': pred_data_atac['cell', 'conn1', 'feat1'].edge_index[1].tolist(),
            'val': all_exist.cpu().detach().numpy()
        })
        matrix_df_zero = d.pivot(index='row', columns='col', values='val')
        matrix_df_zero.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod2_name}-exist.csv")
        
        # Save Transformer Output
        if 'transformer' in pred_atac and 'cell-conn1-feat1' in pred_atac['transformer']:
             pd.DataFrame(pred_atac['transformer']['cell-conn1-feat1']).to_csv(
                f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod2_name}-exist-transformer.csv", sep="\t"
             )
    else:
        # Standard Regression (non-zero-inflated)
        all_pred = pred_atac['cell-conn1-feat1'].reshape(-1)
        d = pd.DataFrame({
            'row': pred_data_atac['cell', 'conn1', 'feat1'].edge_index[0].tolist(), 
            'col': pred_data_atac['cell', 'conn1', 'feat1'].edge_index[1].tolist(),
            'val': all_pred.cpu().detach().numpy()
        })
        matrix_df = d.pivot(index='row', columns='col', values='val')
        matrix_df.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod2_name}-weight.csv")

    # Save Real ATAC Data
    obs_atac.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-obs-{mod2_name}.csv")


    # --- Prediction 2: Predict RNA (mod1) using ATAC cell embeddings ---
    print(f"Predicting {mod1_name} from {mod2_name} embeddings...")
    
    # Slice embeddings for ATAC cells (N_rna to End)
    atac_cell_emb = cell_embeddings[rna_train_shape:(rna_train_shape + atac_train_shape)]
    
    # Prepare prediction data object for RNA features ('conn0', 'feat0')
    pred_data_rna = pred_from_DF(atac_cell_emb, target_rna, zero1, 'conn0', 'feat0')
    pred_data_rna['cell'].x = pred_data_rna['cell'].x.to(torch.float32)
    pred_data_rna = pred_data_rna.to("cpu")
    
    # Run prediction
    pred_rna = predict(model, pred_data_rna)

    # Process and save RNA predictions
    if zero1 == 1:
        all_pred = pred_rna['cell-conn0-feat0'][1].reshape(-1)
        all_exist = pred_rna['cell-conn0-feat0'][0].reshape(-1)
        
        # Save Weights
        d = pd.DataFrame({
            'row': pred_data_rna['cell', 'conn0', 'feat0'].edge_index[0].tolist(), 
            'col': pred_data_rna['cell', 'conn0', 'feat0'].edge_index[1].tolist(),
            'val': all_pred.cpu().detach().numpy()
        })
        matrix_df = d.pivot(index='row', columns='col', values='val')
        matrix_df.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod1_name}-weight.csv")
        
        # Save Existence Probabilities
        d = pd.DataFrame({
            'row': pred_data_rna['cell', 'conn0', 'feat0'].edge_index[0].tolist(), 
            'col': pred_data_rna['cell', 'conn0', 'feat0'].edge_index[1].tolist(),
            'val': all_exist.cpu().detach().numpy()
        })
        matrix_df_zero = d.pivot(index='row', columns='col', values='val')
        matrix_df_zero.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod1_name}-exist.csv")
        
        # Save Transformer Output
        if 'transformer' in pred_rna and 'cell-conn0-feat0' in pred_rna['transformer']:
            pd.DataFrame(pred_rna['transformer']['cell-conn0-feat0']).to_csv(
                f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod1_name}-exist-transformer.csv", sep="\t"
            )
    else:
        all_pred = pred_rna['cell-conn0-feat0'].reshape(-1)
        d = pd.DataFrame({
            'row': pred_data_rna['cell', 'conn0', 'feat0'].edge_index[0].tolist(), 
            'col': pred_data_rna['cell', 'conn0', 'feat0'].edge_index[1].tolist(),
            'val': all_pred.cpu().detach().numpy()
        })
        matrix_df = d.pivot(index='row', columns='col', values='val')
        matrix_df.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-pred-{mod1_name}-weight.csv")

    # Save Real RNA Data
    obs_rna.to_csv(f"{output_dir}/New_{dataset_name}-Set1-GLEAM-obs-{mod1_name}.csv")
    
    print("All predictions saved successfully.")

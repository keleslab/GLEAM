import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class WeightPredictorTransformer(nn.Module):
    """
    A Transformer-based regressor.
    
    It takes a cell embedding vector, projects it into a latent space, processes it 
    with a Transformer Encoder, and outputs a full feature vector (e.g., predicting 
    expression for all genes simultaneously).
    """
    def __init__(self, input_dim, hidden_dim, out_dim, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        # Project input embedding to transformer hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Standard Transformer Encoder Layer
        # batch_first=True means input shape is (batch, seq_len, features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim, # Standard expansion factor of 4
            dropout=dropout,
            activation='gelu', # GELU is often preferred over ReLU in Transformers
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output Head
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim) # Project to final output size (e.g. number of genes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_embeddings):
        # 1. Input Projection
        x = self.norm1(self.input_proj(cell_embeddings))
        
        # 2. Reshape for Transformer
        # Transformer expects a sequence dimension. Since we are processing single cell embeddings,
        # we treat them as a sequence of length 1: (batch_size, 1, hidden_dim)
        x = x.unsqueeze(1)
        
        # 3. Apply Transformer
        x = self.transformer(x)
        
        # 4. Remove sequence dimension
        x = x.squeeze(1)   
        
        # 5. MLP Head for final prediction
        x = self.norm2(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerModel(torch.nn.Module):
    """
    Wrapper class for the WeightPredictorTransformer.
    This simplifies the interface for training/prediction.
    """
    def __init__(self, data, cell_embeddings, input_channels, hidden_channels):
        super().__init__()
        # Instantiate the core transformer module
        # out_dim is set to data.shape[1] (total features to predict)
        self.transformers = WeightPredictorTransformer(
                input_dim=input_channels,
                hidden_dim=hidden_channels,
                out_dim=data.shape[1]
            )
        
    def forward(self, x):
        # Detach input embeddings to prevent gradient flow back to the GNN/Encoder
        # This treats the input embeddings as fixed features for this stage.
        x = x.detach().clone()
        return self.transformers(x)
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.transformers(x)


class TransformerTrainer:
    """
    Dedicated trainer for the standalone TransformerModel.
    Uses Mixed Precision (GradScaler) for efficiency.
    """
    def __init__(self, model, cell_embedding, data, input_channels, hidden_channels=512, lr=0.001, epochs=500):
        # Initialize model
        self.model = model(data, cell_embedding, input_channels, hidden_channels)
        
        # Setup Device (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Inputs and Targets
        self.cell_embedding = cell_embedding
        self.data = data # Target data (Ground Truth)
        self.epochs = epochs
        
        # Mixed Precision Scaler
        self.scaler = GradScaler()

    def train_step(self):
        self.model.train()
        
        # zero_grad(set_to_none=True) is slightly more efficient than set_to_zero
        self.optimizer.zero_grad(set_to_none=True)
        
        # Automatic Mixed Precision Context
        with autocast():
            # Forward pass
            preds = self.model(self.cell_embedding)
            
            # Loss Calculation (MSE Loss between prediction and ground truth)
            # Flattening both tensors ensures shape matching
            loss = F.mse_loss(preds.reshape(-1), self.data.reshape(-1))
            
        # Backward pass with scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss)

    def train(self):
        # Main Training Loop
        for epoch in range(1, self.epochs + 1):
            loss = self.train_step()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
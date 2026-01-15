"""
GLEAM: Graph-based method for multi-modal single-cell integration and
missing-modality prediction.

Public API is organized into:
- embedding / metapath2vec utilities
- core GNN models and training
- preprocessing utilities
- transformer-based predictors
"""

from .enhanceEmb import (
    WeightedMetaPath2Vec,
    weighted_metapath2vec,
    create_adjacency_matrix,
    perform_pca,
    create_knn_graph,
    process_bipartite_graph,
    integrate_data_with_metapath2vec,
)

from .model import (
    CustomGraphConv,
    Encoder,
    EdgePredictor,
    WeightPredictor,
    WeightPredictorTransformer,
    RegressorHybrid,
    Model,
    Trainer,
    predict,
)

from .preprocessing import (
    generate_adjacency_matrix,
    generate_imputed_matrix,
    calculate_tech_zero_proportion,
    load_from_DF,
    pred_from_DF,
    find_kNN,
)

from .transformer import (
    WeightPredictorTransformer as TransformerWeightPredictor,
    TransformerModel,
    TransformerTrainer,
)

__all__ = [
    # enhanceEmb
    "WeightedMetaPath2Vec",
    "weighted_metapath2vec",
    "create_adjacency_matrix",
    "perform_pca",
    "create_knn_graph",
    "process_bipartite_graph",
    "integrate_data_with_metapath2vec",
    # model
    "CustomGraphConv",
    "Encoder",
    "EdgePredictor",
    "WeightPredictor",
    "WeightPredictorTransformer",
    "RegressorHybrid",
    "Model",
    "Trainer",
    "predict",
    # preprocessing
    "generate_adjacency_matrix",
    "generate_imputed_matrix",
    "calculate_tech_zero_proportion",
    "load_from_DF",
    "pred_from_DF",
    "find_kNN",
    # transformer
    "TransformerWeightPredictor",
    "TransformerModel",
    "TransformerTrainer",
]
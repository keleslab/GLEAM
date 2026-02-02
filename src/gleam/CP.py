import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, List, Optional, Dict, Any

# Optional: Handle AnnData if installed, otherwise treat as generic object
try:
    from anndata import AnnData
except ImportError:
    AnnData = None

def run_gleam_conformal(
    pred_val: Union[np.ndarray, pd.DataFrame, Any],
    pred_prob: Union[np.ndarray, pd.DataFrame, Any],
    true_val: Union[np.ndarray, pd.DataFrame, Any],
    calib_pred_val: Optional[Union[np.ndarray, pd.DataFrame, Any]] = None,
    calib_pred_prob: Optional[Union[np.ndarray, pd.DataFrame, Any]] = None,
    calib_true_val: Optional[Union[np.ndarray, pd.DataFrame, Any]] = None,
    alpha: float = 0.33,
    num_categories: int = 4
) -> Dict[str, Any]:
    """
    Main wrapper for GLEAM Two-Part Conformal Prediction.
    
    Parameters:
    -----------
    pred_val : Predicted expression values (Test set).
    pred_prob : Predicted probability of being ZERO (Test set).
    true_val : True expression values (Test set).
    calib_* : Corresponding arrays for the calibration set.
    alpha : Error tolerance (default 0.33).
    num_categories : Number of sparsity bins for stratified calibration.
    
    Returns:
    --------
    Dictionary containing 'classification' results and 'regression' dataframes.
    """
    
    # 1. Standardize Inputs to Dense Numpy Arrays
    X_pred = _process_input(pred_val)
    P_zero = _process_input(pred_prob)
    Y_true = _process_input(true_val)
    
    has_calib = (calib_pred_val is not None) and (calib_true_val is not None)
    
    if has_calib:
        X_calib = _process_input(calib_pred_val)
        P_calib = _process_input(calib_pred_prob)
        Y_calib = _process_input(calib_true_val)
    else:
        X_calib, P_calib, Y_calib = None, None, None

    # 2. Part A: Conformal Classification (Zero vs Non-Zero)
    # We flatten inputs because classification coverage is element-wise independent
    cls_res = get_conformal_classification(
        x_pred_prob=P_zero.ravel(),
        x_true=Y_true.ravel(),
        c_pred_prob=P_calib.ravel() if has_calib else None,
        c_true=Y_calib.ravel() if has_calib else None,
        alpha=alpha
    )
    
    # 3. Part B: Conformal Regression (Stratified Non-Zero Intervals)
    # We pass full matrices to calculate gene-wise sparsity correctly
    reg_res = get_stratified_intervals(
        x_pred=X_pred,
        x_prob_zero=P_zero,
        x_true=Y_true,
        c_pred=X_calib,
        c_prob_zero=P_calib,
        c_true=Y_calib,
        alpha=alpha,
        num_categories=num_categories
    )
    
    return {
        "classification": cls_res,
        "regression": reg_res
    }

def get_conformal_classification(
    x_pred_prob: np.ndarray, 
    x_true: np.ndarray, 
    c_pred_prob: Optional[np.ndarray] = None, 
    c_true: Optional[np.ndarray] = None, 
    alpha: float = 0.33
) -> Dict[str, Any]:
    """
    Calculates conformal sets for binary classification (Zero vs Non-Zero).
    Uses 'Conformity Score' logic: Score = Probability of the true class.
    """
    
    # Calculate scores on calibration set if available
    if c_pred_prob is not None and c_true is not None:
        # If True=0, Score is Prob(0). If True!=0, Score is Prob(Non-Zero) = 1-Prob(0)
        # Assuming input c_pred_prob is Probability of ZERO.
        calib_scores = np.where(c_true == 0, c_pred_prob, 1.0 - c_pred_prob)
        
        # We want to cover (1-alpha) of the mass.
        # Since these are conformity scores (high is good), we reject the bottom alpha.
        q = np.nanquantile(calib_scores, alpha)
    else:
        # Fallback to test set (not recommended, provided for logic continuity)
        scores = np.where(x_true == 0, x_pred_prob, 1.0 - x_pred_prob)
        q = np.nanquantile(scores, alpha)

    # Construct Sets: Include class if Prob(class) >= q
    include_zero = x_pred_prob >= q
    include_pos = (1.0 - x_pred_prob) >= q
    
    # Combine into a mask (N_samples x 2 classes)
    # Column 0 is "Is Zero in set?", Column 1 is "Is Non-Zero in set?"
    sets = np.column_stack((include_zero, include_pos))
    set_sizes = sets.sum(axis=1)
    
    # Calculate coverage: Was the true label inside the generated set?
    covered = (
        ((x_true == 0) & include_zero) | 
        ((x_true > 0) & include_pos)
    )
    
    return {
        "conformal_sets": sets,
        "coverage": np.mean(covered),
        "average_set_size": np.mean(set_sizes),
        "threshold": q
    }

def get_stratified_intervals(
    x_pred: np.ndarray,
    x_prob_zero: np.ndarray,
    x_true: np.ndarray,
    c_pred: Optional[np.ndarray] = None,
    c_prob_zero: Optional[np.ndarray] = None,
    c_true: Optional[np.ndarray] = None,
    alpha: float = 0.33,
    num_categories: int = 4
) -> Optional[pd.DataFrame]:
    """
    Calculates regression prediction intervals for non-zero values,
    stratified by gene sparsity levels.
    """
    
    # Sanitize inputs
    x_pred = np.maximum(x_pred, 0)
    if c_pred is not None:
        c_pred = np.maximum(c_pred, 0)

    # Convert Prob(Zero) to Prob(NonZero) for scaling
    x_prob_nz = 1.0 - x_prob_zero
    
    if c_pred is None or c_true is None:
        print("No calibration data provided for intervals. Returning None.")
        return None
        
    c_prob_nz = 1.0 - c_prob_zero

    # --- 1. Identify Non-Zero Indices in Test Data ---
    nz_mask = x_true > 0
    if not np.any(nz_mask):
        return None

    # --- 2. Calculate Gene Sparsity based on Calibration Data ---
    # Assume columns are genes. If 1D, treat as single gene.
    if c_true.ndim > 1:
        # Sparsity = fraction of zeros per gene (column)
        gene_sparsity = np.mean(c_true == 0, axis=0)
        n_genes = c_true.shape[1]
        n_samples_test = x_true.shape[0]
        
        # Create vector of gene indices matching flattened array
        # Flatten order in numpy is usually row-major (C-style), 
        # but to match column-wise logic easily we need to be careful.
        # Let's stick to simple flattening: row 0 all genes, row 1 all genes...
        # So gene index pattern is [0, 1, 2... 0, 1, 2...]
        gene_indices_flat = np.tile(np.arange(n_genes), n_samples_test)
        
        # We must align other arrays to this flattening
        # Note: .ravel() flattens row by row by default (C-order)
    else:
        gene_sparsity = np.array([np.mean(c_true == 0)])
        gene_indices_flat = np.zeros(x_true.size, dtype=int)

    # --- 3. Bin Genes by Sparsity ---
    # Handle unique breaks logic
    try:
        # qcut tries to divide into equal sized bins based on quantiles
        gene_cats = pd.qcut(gene_sparsity, q=num_categories, labels=False, duplicates='drop')
        # If duplicates dropped, we might have fewer categories than requested
        actual_num_cats = gene_cats.max() + 1
    except ValueError:
        # Fallback if mostly constant
        gene_cats = np.zeros_like(gene_sparsity, dtype=int)
        actual_num_cats = 1

    # --- 4. Calculate Calibration Quantiles per Category ---
    
    # Flatten calibration arrays
    flat_c_true = c_true.ravel()
    flat_c_pred = c_pred.ravel()
    flat_c_prob = c_prob_nz.ravel()
    
    # Map calibration flattened indices to gene categories
    n_samples_calib = c_true.shape[0]
    if c_true.ndim > 1:
        c_gene_indices = np.tile(np.arange(c_true.shape[1]), n_samples_calib)
    else:
        c_gene_indices = np.zeros(c_true.size, dtype=int)
        
    c_obs_cats = gene_cats[c_gene_indices]

    # Filter for non-zero calibration points
    c_nz_mask = flat_c_true > 0
    
    # Calculate errors: |y - y_hat| / prob_nz
    # Epsilon to prevent division by zero
    c_errors = np.abs(flat_c_true - flat_c_pred) / (flat_c_prob + 1e-9)
    
    # Fallback global quantile
    fallback_q = np.nanquantile(c_errors[c_nz_mask], 1.0 - alpha)
    
    category_qs = {}
    
    for cat in range(actual_num_cats):
        # Mask: belongs to category AND is non-zero
        cat_mask = (c_obs_cats == cat) & c_nz_mask
        
        if np.sum(cat_mask) < 5:
            category_qs[cat] = fallback_q
        else:
            category_qs[cat] = np.nanquantile(c_errors[cat_mask], 1.0 - alpha)

    # --- 5. Apply to Test Data ---
    
    # Flatten Test Data
    flat_x_pred = x_pred.ravel()
    flat_x_prob = x_prob_nz.ravel()
    flat_x_true = x_true.ravel()
    flat_nz_mask = flat_x_true > 0
    
    # Extract only non-zero test instances
    x_pred_nz = flat_x_pred[flat_nz_mask]
    x_prob_nz_vec = flat_x_prob[flat_nz_mask]
    x_true_nz = flat_x_true[flat_nz_mask]
    
    # Get categories for these test instances
    nz_gene_indices = gene_indices_flat[flat_nz_mask]
    nz_cats = gene_cats[nz_gene_indices]
    
    # Map categories to Q values
    # Use list comprehension or vector map
    obs_qs = np.array([category_qs.get(c, fallback_q) for c in nz_cats])
    
    # Calculate Bounds
    lower = x_pred_nz - (obs_qs * x_prob_nz_vec)
    upper = x_pred_nz + (obs_qs * x_prob_nz_vec)
    
    lower = np.maximum(lower, 0)
    
    return pd.DataFrame({
        'predicted': x_pred_nz,
        'lower': lower,
        'upper': upper,
        'true': x_true_nz,
        'category': nz_cats
    })

def _process_input(obj: Any) -> Optional[np.ndarray]:
    """Helper to convert diverse inputs (AnnData, DF, List) to dense Numpy Array."""
    if obj is None:
        return None
    
    # Handle AnnData
    if AnnData is not None and isinstance(obj, AnnData):
        if sparse.issparse(obj.X):
            return obj.X.toarray()
        return obj.X
        
    # Handle DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.values
        
    # Handle Sparse Matrix
    if sparse.issparse(obj):
        return obj.toarray()
        
    # Handle standard Lists/Tuples
    if isinstance(obj, (list, tuple)):
        return np.array(obj)
        
    # Already Numpy
    if isinstance(obj, np.ndarray):
        return obj
        
    # Fallback
    return np.array(obj)
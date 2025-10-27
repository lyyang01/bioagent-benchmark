# MOFA - Multi-Omics Factor Analysis

## Overview

MOFA is a method for integrating multiple modalities of omics data through unsupervised dimensionality reduction. It identifies the principal sources of variation across data modalities and learns a low-dimensional representation of the data.

## Function
import muon as mu

```python
muon.tl.mofa(
    data: AnnData | MuData, 
    groups_label: bool = None, 
    use_raw: bool = False, 
    use_layer: str = None, 
    use_var: str | None = 'highly_variable', 
    use_obs: str | None = None, 
    likelihoods: str | List[str] | None = None, 
    n_factors: int = 10, 
    scale_views: bool = False, 
    scale_groups: bool = False, 
    center_groups: bool = True, 
    ard_weights: bool = True, 
    ard_factors: bool = True, 
    spikeslab_weights: bool = True, 
    spikeslab_factors: bool = False, 
    n_iterations: int = 1000, 
    convergence_mode: str = 'fast', 
    use_float32: bool = False, 
    gpu_mode: bool = False, 
    gpu_device: bool | None = None, 
    svi_mode: bool = False, 
    svi_batch_size: float = 0.5, 
    svi_learning_rate: float = 1.0, 
    svi_forgetting_rate: float = 0.5, 
    svi_start_stochastic: int = 1, 
    smooth_covariate: str | None = None, 
    smooth_warping: bool = False, 
    smooth_kwargs: Mapping[str, Any] | None = None, 
    save_parameters: bool = False, 
    save_data: bool = True, 
    save_metadata: bool = True, 
    seed: int = 1, 
    outfile: str | None = None, 
    expectations: List[str] | None = None, 
    save_interrupted: bool = True, 
    verbose: bool = False, 
    quiet: bool = True, 
    copy: bool = False
)
```

Run Multi-Omics Factor Analysis on single-cell data.

## Parameters

### Input Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `AnnData \| MuData` | required | An AnnData or MuData object containing the multi-omic data to analyze |
| `groups_label` | `bool` | `None` | A column name in adata.obs for grouping the samples |
| `use_raw` | `bool` | `False` | Use raw slot of AnnData as input values |
| `use_layer` | `str` | `None` | Use a specific layer of AnnData as input values (supersedes use_raw option) |
| `use_var` | `str \| None` | `'highly_variable'` | .var column with a boolean value to select genes (e.g. "highly_variable") |
| `use_obs` | `str \| None` | `None` | Strategy to deal with samples (cells) not being the same across modalities ("union" or "intersection", throw error by default) |
| `likelihoods` | `str \| List[str] \| None` | `None` | Likelihoods to use, default is guessed from the data |

### Model Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_factors` | `int` | `10` | Number of factors to train the model with |
| `scale_views` | `bool` | `False` | Scale views to unit variance |
| `scale_groups` | `bool` | `False` | Scale groups to unit variance |
| `center_groups` | `bool` | `True` | Center groups to zero mean |
| `ard_weights` | `bool` | `True` | Use view-wise sparsity |
| `ard_factors` | `bool` | `True` | Use group-wise sparsity |
| `spikeslab_weights` | `bool` | `True` | Use feature-wise sparsity (e.g. gene-wise) |
| `spikeslab_factors` | `bool` | `False` | Use sample-wise sparsity (e.g. cell-wise) |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iterations` | `int` | `1000` | Upper limit on the number of iterations |
| `convergence_mode` | `str` | `'fast'` | Fast, medium, or slow convergence mode |
| `use_float32` | `bool` | `False` | Use reduced precision (float32) |
| `gpu_mode` | `bool` | `False` | If to use GPU mode |
| `gpu_device` | `bool \| None` | `None` | Which GPU device to use |
| `seed` | `int` | `1` | Random seed |

### Stochastic Variational Inference Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `svi_mode` | `bool` | `False` | If to use Stochastic Variational Inference (SVI) |
| `svi_batch_size` | `float` | `0.5` | Batch size as a fraction (only applicable when svi_mode=True) |
| `svi_learning_rate` | `float` | `1.0` | Learning rate (only applicable when svi_mode=True) |
| `svi_forgetting_rate` | `float` | `0.5` | Forgetting rate (only applicable when svi_mode=True) |
| `svi_start_stochastic` | `int` | `1` | First iteration to start SVI (only applicable when svi_mode=True) |

### MEFISTO Smoothing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smooth_covariate` | `str \| None` | `None` | Use a covariate (column in .obs) to learn smooth factors (MEFISTO) |
| `smooth_warping` | `bool` | `False` | If to learn the alignment of covariates (e.g. time points) from different groups; by default, the first group is used as a reference |
| `smooth_kwargs` | `Mapping[str, Any] \| None` | `None` | Additional arguments for MEFISTO (covariates_names, scale_cov, start_opt, n_grid, opt_freq, warping_freq, warping_ref, warping_open_begin, warping_open_end, sparseGP, frac_inducing, model_groups, new_values) |

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_parameters` | `bool` | `False` | If to save training parameters |
| `save_data` | `bool` | `True` | If to save training data |
| `save_metadata` | `bool` | `True` | If to load metadata from the AnnData object (.obs and .var tables) and save it |
| `outfile` | `str \| None` | `None` | Path to HDF5 file to store the model |
| `expectations` | `List[str] \| None` | `None` | Which nodes should be used to save expectations for (will save only W and Z by default). Possible expectations include Y, W, Z, Tau, AlphaZ, AlphaW, ThetaW, ThetaZ |
| `save_interrupted` | `bool` | `True` | If to save partially trained model when the training is interrupted |

### Miscellaneous Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | Print verbose information during training |
| `quiet` | `bool` | `True` | Silence messages during training procedure |
| `copy` | `bool` | `False` | Return a copy of AnnData instead of writing to the provided object |

## Returns

Returns or updates the input data object with MOFA factors and weights. If `copy=True`, returns a copy of the AnnData object with the MOFA results.

## Notes

- MOFA can be used for multi-modal integration of single-cell data, including RNA-seq, ATAC-seq, protein measurements, etc.
- The results are stored in the AnnData object under `.obsm['X_mofa']` for the factors, and `.varm['mofa']` for the weights.
- The model can also be saved to an HDF5 file for later use by providing the `outfile` parameter.

## Example

```python
import muon as mu
import scanpy as sc

# Load data
adata = sc.read_h5ad("rna_data.h5ad")

# Preprocess data
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Run MOFA
mu.tl.mofa(
    adata,
    use_var='highly_variable',
    n_factors=15,
    center_groups=True,
    scale_views=True,
    n_iterations=1000,
    convergence_mode='medium',
    save_metadata=True
)

# Plot MOFA factors
sc.pl.embedding(adata, basis='X_mofa', color=['cell_type'])
```
# Hotspot
## Overview

Hotspot is a tool for analyzing gene expression data to identify spatial patterns of gene expression and create gene modules based on local correlations.

## Class: Hotspot

```python
hotspot.Hotspot(
    adata, 
    layer_key=None, 
    model='danb', 
    latent_obsm_key=None, 
    distances_obsp_key=None, 
    tree=None, 
    umi_counts_obs_key=None
)
```

Initialize a Hotspot object for analysis.

**Note**: Either `latent_obsm_key`, `distances_obsp_key`, or `tree` is required.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | required | Count matrix (shape is cells by genes) |
| `layer_key` | `str` | `None` | Key in adata.layers with count data, uses adata.X if None |
| `model` | `str` | `'danb'` | Null model for gene expression: 'danb' (Depth-Adjusted Negative Binomial), 'bernoulli' (Models probability of detection), 'normal' (Depth-Adjusted Normal), or 'none' (Assumes data has been pre-standardized) |
| `latent_obsm_key` | `str` | `None` | Key in adata.obsm for latent space encoding cell-cell similarities with euclidean distances (cells x dims) |
| `distances_obsp_key` | `str` | `None` | Key in adata.obsp for distances encoding cell-cell similarities directly (cells x cells) |
| `tree` | `ete3.coretype.tree.TreeNode` | `None` | Root tree node (can be created using ete3.Tree) |
| `umi_counts_obs_key` | `str` | `None` | Total UMI count per cell (used as a size factor). If omitted, the sum over genes in the counts matrix is used |

## Methods

### legacy_init

```python
@classmethod
hotspot.Hotspot.legacy_init(
    counts, 
    model='danb', 
    latent=None, 
    distances=None, 
    tree=None, 
    umi_counts=None
)
```

Initialize a Hotspot object using the legacy method.

**Note**: Either `latent`, `distances`, or `tree` is required.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `counts` | `pandas.DataFrame` | required | Count matrix (shape is genes x cells) |
| `model` | `str` | `'danb'` | Null model for gene expression |
| `latent` | `pandas.DataFrame` | `None` | Latent space encoding cell-cell similarities (cells x dims) |
| `distances` | `pandas.DataFrame` | `None` | Distances encoding cell-cell similarities directly (cells x cells) |
| `tree` | `ete3.coretype.tree.TreeNode` | `None` | Root tree node |
| `umi_counts` | `pandas.Series` | `None` | Total UMI count per cell |



### create_knn_graph

```python
create_knn_graph(
    weighted_graph=False, 
    n_neighbors=30, 
    neighborhood_factor=3, 
    approx_neighbors=True
)
```

Create the KNN graph and graph weights.

The resulting matrices containing the neighbors and weights are stored in the object at `self.neighbors` and `self.weights`.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weighted_graph` | `bool` | `False` | Whether or not to create a weighted graph |
| `n_neighbors` | `int` | `30` | Neighborhood size |
| `neighborhood_factor` | `float` | `3` | Used for weighted graphs; controls decay rate of weights with distance |
| `approx_neighbors` | `bool` | `True` | Use approximate nearest neighbors (only when initialized with latent) |

### compute_autocorrelations

```python
compute_autocorrelations(jobs=1)
```

Perform feature selection using local autocorrelation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jobs` | `int` | `1` | Number of parallel jobs to run |

#### Returns

| Return Type | Description |
|-------------|-------------|
| `pandas.DataFrame` | A dataframe with columns: 'C' (Scaled -1:1 autocorrelation coefficients), 'Z' (Z-score), 'Pval' (P-values from Z-scores), and 'FDR' (Q-values using Benjamini-Hochberg procedure). Gene IDs are in the index. |

**Note**: Results are also stored in `self.results`.

### compute_local_correlations

```python
compute_local_correlations(genes, jobs=1)
```

Define gene-gene relationships with pair-wise local correlations.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genes` | `iterable of str` | required | Gene identifiers to compute local correlations on (should be a smaller subset of all genes) |
| `jobs` | `int` | `1` | Number of parallel jobs to run |

#### Returns

| Return Type | Description |
|-------------|-------------|
| `pandas.DataFrame` | Local correlation Z-scores between genes (shape is genes x genes) |

**Note**: Results are also stored in `self.local_correlation_z`.

### create_modules

```python
create_modules(min_gene_threshold=20, core_only=True, fdr_threshold=0.05)
```

Group genes into modules.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_gene_threshold` | `int` | `20` | Controls minimum module size |
| `core_only` | `bool` | `True` | Whether to assign ambiguous genes to a module or leave unassigned |
| `fdr_threshold` | `float` | `0.05` | Correlation threshold for module assignment |

#### Returns

| Return Type | Description |
|-------------|-------------|
| `pandas.Series` | Maps gene to module number. Unassigned genes are indicated with -1 |

**Note**: Results are stored in `self.modules` and the linkage matrix is saved in `self.linkage`.

### calculate_module_scores

```python
calculate_module_scores()
```

Calculate module scores.

#### Returns

| Return Type | Description |
|-------------|-------------|
| `pandas.DataFrame` | Scores for each module for each gene (dimensions are genes x modules) |

**Note**: Results are stored in `self.module_scores`.

### plot_local_correlations

```python
plot_local_correlations(
    mod_cmap='tab10', 
    vmin=-8, 
    vmax=8, 
    z_cmap='RdBu_r', 
    yticklabels=False
)
```

Plot a clustergrid of the local correlation values.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mod_cmap` | `str or colormap` | `'tab10'` | Discrete colormap for module assignments |
| `vmin` | `float` | `-8` | Minimum value for Z-score colorscale |
| `vmax` | `float` | `8` | Maximum value for Z-score colorscale |
| `z_cmap` | `str or colormap` | `'RdBu_r'` | Continuous colormap for correlation Z-scores |
| `yticklabels` | `bool` | `False` | Whether to plot all gene labels |


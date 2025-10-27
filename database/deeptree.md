# DeepTree

## Overview

DeepTree is a feature selection algorithm that improves single-cell RNA-seq analysis by removing noise from highly variable genes. It mitigates batch effects, increases the signal-to-noise ratio of principal components, and promotes the discovery of rare cell types.

## Function

```python
Scanpyplus.DeepTree(
    adata,
    MouseC1ColorDict2,
    cell_type='louvain',
    gene_type='highly_variable',
    cellnames=['default'],
    genenames=['default'],
    figsize=(10,7),
    row_cluster=True,
    col_cluster=True,
    method='complete',
    metric='correlation',
    Cutoff=0.8,
    CladeSize=2
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | required | Annotated data matrix with single-cell data |
| `MouseC1ColorDict2` | `dict` | required | Dictionary specifying colors for row labels, e.g., `{False:'#000000', True:'#00FFFF'}` |
| `cell_type` | `str` | `'louvain'` | Column name in `adata.obs` that contains cell type/cluster annotations |
| `gene_type` | `str` | `'highly_variable'` | Column name in `adata.var` that specifies gene selection, typically 'highly_variable' |
| `cellnames` | `list` | `['default']` | List of cell names to include in the analysis. If `'default'`, uses all cells in `adata.obs_names` |
| `genenames` | `list` | `['default']` | List of gene names to include in the analysis. If `'default'`, uses all genes in `adata.var_names` |
| `figsize` | `tuple` | `(10,7)` | Figure size for the output heatmaps |
| `row_cluster` | `bool` | `True` | Whether to cluster rows (genes) in the heatmap |
| `col_cluster` | `bool` | `True` | Whether to cluster columns (cells) in the heatmap |
| `method` | `str` | `'complete'` | Linkage method for hierarchical clustering, passed to the underlying `scipy.cluster.hierarchy` function |
| `metric` | `str` | `'correlation'` | Distance metric used for clustering, passed to the underlying `scipy.cluster.hierarchy` function |
| `Cutoff` | `float` | `0.8` | Height threshold for cutting the hierarchical clustering tree |
| `CladeSize` | `int` | `2` | Minimum size of clades to consider for the filtered gene set |

## Returns

A list containing the following elements:

| Index | Type | Description |
|-------|------|-------------|
| 0 | `AnnData` | Filtered AnnData object with selected genes. A new column `'Deep'` is added to `adata.var`, indicating which genes are retained |
| 1 | `ClusterGrid` | Initial clustering result on the input data |
| 2 | `ClusterGrid` | Clustering result after marking genes with the 'Deep' annotation |
| 3 | `ClusterGrid` | Final clustering result using only the filtered 'Deep' genes |

## Algorithm

1. Performs hierarchical clustering on the input data (typically using highly variable genes)
2. Cuts the hierarchical tree at the specified height (`Cutoff`)
3. Identifies gene clusters (clades) larger than the specified size (`CladeSize`)
4. Creates a filtered gene set from these significant clades
5. Annotates the selected genes in a new `'Deep'` column in `adata.var`
6. Returns visualizations of the different clustering stages

## Example

```python
import Scanpyplus

# Generate a color dictionary for cell types
cell_colors = {
    'Type A': '#FF0000',
    'Type B': '#00FF00',
    'Type C': '#0000FF'
}

# Create a mapping from cells to colors
color_dict = {
    cell: cell_colors[adata.obs['leiden'][cell]] 
    for cell in adata.obs_names
}

# Run DeepTree
[filtered_data, initial_cluster, marked_cluster, final_cluster] = Scanpyplus.DeepTree(
    adata,
    MouseC1ColorDict2=color_dict,
    cell_type='leiden',
    row_cluster=True,
    col_cluster=True
)

# Use the filtered data for downstream analysis
import scanpy as sc
sc.pp.neighbors(filtered_data)
sc.tl.umap(filtered_data)
sc.pl.umap(filtered_data, color='leiden')
```

## Notes

- DeepTree is particularly useful for "soft integration" of data across batches, conditions, or time points
- It can help reduce batch effects while avoiding overcorrection that might remove biological signal
- The algorithm effectively removes "garbage" genes among highly variable genes
- Implementation is also available in MATLAB
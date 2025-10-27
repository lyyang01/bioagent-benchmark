# DouCLing (Doublet Cluster Labeling)

## Overview

DouCLing is a method for identifying potential doublet clusters in single-cell RNA sequencing data. The algorithm visits each subcluster and tries to find its two parent cell types:
- **Parent 1**: The initial annotation (the bigger cluster the subcluster belongs to)
- **Parent 2**: Another bigger cluster that explains the unique signatures of this subcluster that aren't shared with its sibling subclusters

## Description

The algorithm works by:
1. Calculating marker genes for each subcluster relative to its sibling subclusters
2. Using these marker genes to score all cells in the dataset
3. Identifying "source" cells (cells with higher scores than the average of the subcluster)
4. Determining if a significant fraction of source cells come from a specific big cluster (Parent 2)
5. Labeling subclusters as doublets when more than `fraction_threshold` of source cells are from Parent 2

## Function

```python
Scanpyplus.DouCLing(
    adata,
    hi_type,
    lo_type,
    rm_genes=[],
    print_marker_genes=False,
    fraction_threshold=0.6
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | `AnnData` | Annotated data matrix object containing single-cell data |
| `hi_type` | `str` | Key for high-resolution cell types in `adata.obs` that may include doublet clusters (e.g., 'Celltype_R') |
| `lo_type` | `str` | Key for low-resolution cell types in `adata.obs` that represent major compartments (e.g., 'Celltype') |
| `rm_genes` | `list`, optional | List of genes to exclude from analysis (e.g., cell-cycle genes). Default: `[]` |
| `print_marker_genes` | `bool`, optional | Whether to print marker genes used for scoring. Default: `False` |
| `fraction_threshold` | `float`, optional | Threshold for determining doublet clusters. Default: `0.6` |

### Returns

| Return Type | Description |
|-------------|-------------|
| `pandas.DataFrame` | Results table with following columns:<br>- **Parent1**: The primary cell type annotation<br>- **Parent2**: The secondary cell type that contributes to the cluster<br>- **Parent1_count**: Number of source cells from Parent1<br>- **Parent2_count**: Number of source cells from Parent2<br>- **All_count**: Total number of source cells<br>- **p_value**: Hypergeometric test p-value for enrichment<br>- **Is_doublet_cluster**: Boolean indicating if cluster is likely a doublet |

## Example Usage

```python
import Scanpyplus

# Run DouCLing with a 0.4 threshold
DoubletReport = Scanpyplus.DouCLing(
    adata,
    hi_type='Celltype_R',
    lo_type='Celltype',
    fraction_threshold=0.4
)
```

## Notes

- This function aims to identify cross-compartment doublet clusters
- Same-compartment doublet clusters (homotypic doublets) are more difficult to identify as they resemble transitional cell states
- The method uses a hypergeometric test to calculate statistical significance
- Performance time is reported upon completion
- Clusters are labeled as doublets when the fraction of Parent2 source cells exceeds `fraction_threshold` AND Parent1 ≠ Parent2

## Algorithm Details

1. For each unique value in `lo_type` (major cell compartments):
   - Extract cells belonging to this compartment
   - Compute marker genes for each subcluster using Scanpy's `rank_genes_groups`
   - For each subcluster:
     - Select top marker genes (excluding genes in `rm_genes`)
     - Score all cells in the dataset using these marker genes
     - Determine Parent1 (the most common `lo_type` in the subcluster)
     - Set score cutoff as the 75th percentile of scores within the subcluster
     - Count cells above cutoff from each major cell type
     - Identify Parent2 (the cell type with highest count above cutoff)
     - Calculate statistics and hypergeometric test p-value
     - Label as doublet if Parent2 contributes more than `fraction_threshold` and is different from Parent1
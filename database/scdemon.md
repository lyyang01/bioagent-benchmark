# scDemon

## Overview

scDemon (single-cell Differentiation Entropy Modules for ONtology) is a tool for calculating gene modules from single-cell data. It uses singular value decomposition (SVD) to identify correlated gene sets that may represent functional modules.

## Main Class: `modules`

```python
import scdemon as sm
mod = sm.modules(adata, suffix=tag, k=max_k, filter_expr=0.05)
```

### Constructor

```python
scdemon.modules(
    adata, 
    U=None, 
    s=None, 
    V=None, 
    suffix='', 
    seed=1, 
    k=100, 
    filter_expr=0.05, 
    keep_first_PC=False, 
    process_covariates=False, 
    covariate_cutoff=0.4
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | required | Single-cell dataset from scanpy |
| `U` | `np.array` | `None` | SVD left singular vectors of size (n_obs, k). If all of U, s, V are not provided, will re-calculate SVD on given data |
| `s` | `np.array` | `None` | SVD singular values of size (k) |
| `V` | `np.array` | `None` | SVD right singular vectors of size (k, n_var) |
| `suffix` | `str` | `''` | Unique suffix for saving image and table filenames |
| `seed` | `int` | `1` | Seed, for np.random.seed |
| `k` | `int` | `100` | Truncate SVD to k components when estimating correlation |
| `filter_expr` | `float` | `0.05` | Remove all genes whose fraction of non-zero values is below the given cutoff |
| `keep_first_PC` | `bool` | `False` | Keep the first PC when estimating the correlation |
| `process_covariates` | `bool` | `False` | Compare the metadata covariates to the PCs |
| `covariate_cutoff` | `float` | `0.4` | Cutoff value for covariate processing |

### Methods

#### setup()

Set up the dataset. Filter genes, calculate PCA, and process covariates. PCA defaults to adata.obsm['X_pca'] if available. Otherwise uses sc.tl.pca from scanpy.

```python
mod.setup()
```

#### make_graph(graph_id, multigraph=False, power=0, **kwargs)

Creates a graph under .graphs[graph_id]

```python
mod.make_graph("my_graph", resolution=1.0, method="bivariate")
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_id` | `str` | required | Unique name for storing/accessing the graph |
| `multigraph` | `bool` | `False` | Create from one graph or multiple graphs |
| `power` | `float \| list` | `0` | Power parameter, either single or multiple or list of powers |
| `method` | `str` | `'bivariate'` | Thresholding method for graph: 'bivariate' (default, threshold based on bivariate spline fit to gene-gene sparsity), 'cutoff' (single cutoff across full matrix), or 'sd' (based on estimated sd, expensive) |
| `filter_covariate` | `str` | `None` | Filter SVD components correlated with a specific covariate |
| `raw` | `bool` | `False` | Use raw correlation |
| `resolution` | `float` | `None` | Resolution for clustering |
| `adjacency_only` | `bool` | `False` | Only run adjacency (default False) |
| `full_graph_only` | `bool` | `False` | Compute the full, unaltered, graph for multiplexing, but do not cluster or create modules |
| `keep_all_z` | `bool` | `False` | When computing full graph, don't threshold, keep dense matrix |
| `layout` | `bool` | `True` | Lay out graph (default True) |
| `**kwargs` | | | Any args for adjacency_matrix or gene_graph |

#### get_k_stats(k_list, power=0, resolution=None, raw=False, method='bivariate', filter_covariate=None, **kwargs)

Get statistics on # genes and # modules for each number of SVD components (k)

```python
ngenes, nmodules = mod.get_k_stats([50, 75, 100])
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_list` | `list` | required | List of SVD component cutoffs |
| `power` | `float` | `0` | Power parameter, here only single value |
| `raw` | `bool` | `False` | Use raw correlation |
| `resolution` | `float` | `None` | Resolution for clustering |
| `method` | `str` | `'bivariate'` | Thresholding method for graph: 'bivariate' (default), 'cutoff', or 'sd' |
| `filter_covariate` | `str` | `None` | Filter SVD components correlated with a specific covariate |
| `**kwargs` | | | Extended arguments for adjacency_matrix or gene_graph |

##### Returns

Lists of number of genes and modules identified at each setting of k

#### recluster_graph(graph_id, resolution=None)

Re-cluster a graph with a different resolution.

```python
mod.recluster_graph("my_graph", resolution=1.5)
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_id` | `str` | required | Unique name for storing/accessing the graph |
| `resolution` | `float` | `None` | Resolution for clustering |

#### get_modules(graph_id, attr='leiden', print_modules=False)

Get list of modules from graph and clustering.

```python
modules_dict = mod.get_modules("my_graph", print_modules=True)
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_id` | `str` | required | Unique name for storing/accessing the graph |
| `attr` | `str` | `'leiden'` | Modules name within the graph ('leiden' is the only current supported method) |
| `print_modules` | `bool` | `False` | Whether to print modules at the same time |

##### Returns

Dictionary of genes in each module in the graph

#### get_module_assignment(graph_id, attr='leiden')

Get module assignment for each gene as a pandas DataFrame.

```python
assignments_df = mod.get_module_assignment("my_graph")
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_id` | `str` | required | Unique name for storing/accessing the graph |
| `attr` | `str` | `'leiden'` | Modules name within the graph ('leiden' is the only current supported method) |

##### Returns

pandas.DataFrame with gene to module assignments

#### find_gene(graph_id, gene, return_genes=True, print_genes=True)

Find the module containing a specific gene.

```python
module_genes = mod.find_gene("my_graph", "GENE1")
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_id` | `str` | required | Unique name for storing/accessing the graph |
| `gene` | `str` | required | Gene to look up in the modules |
| `return_genes` | `bool` | `True` | Whether to return genes in the module |
| `print_genes` | `bool` | `True` | Whether to print genes at the same time |

##### Returns

If return_genes=True, returns the list of genes in the module that contains the gene in question

#### save_modules(graph_id, attr='leiden', as_df=True, filedir='./', filename=None)

Save module list for a specific graph as txt or tsv.

```python
mod.save_modules("my_graph", filedir="./results/")
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_id` | `str` | required | Unique name for storing/accessing the graph |
| `attr` | `str` | `'leiden'` | Modules name within the graph ('leiden' is the only current supported method) |
| `as_df` | `bool` | `True` | Write out dataframe instead of a raw list of genes per module |
| `filedir` | `str` | `'./'` | Directory for file, defaults to ./ |
| `filename` | `str` | `None` | Name for file overriding default naming scheme |

## Example Usage

```python
import scanpy as sc
import scdemon as sm

# Load data
adata = sc.read_h5ad("my_dataset.h5ad")

# Pre-process with scanpy
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

# Initialize scDemon
mod = sm.modules(adata, suffix="example", k=100, filter_expr=0.05)

# Setup and create graph
mod.setup()
mod.make_graph("main_graph", method="bivariate", resolution=1.0)

# Get modules
modules_dict = mod.get_modules("main_graph", print_modules=True)

# Save results
mod.save_modules("main_graph", filedir="./results/")

# Find a specific gene
mod.find_gene("main_graph", "GENE1")
```

## Notes

- scDemon uses SVD and graph-based clustering to identify gene modules in single-cell data
- Leiden community detection is used for clustering the gene-gene graph
- The method creates a graph based on gene-gene correlations in reduced SVD space
- Multiple graph creation methods are supported with the `method` parameter
- The `k` parameter (number of SVD components) is important for determining the granularity of modules
# Squidpy
## spatial_neighbors
import squidpy 

```python
squidpy.gr.spatial_neighbors(
    adata, 
    spatial_key='spatial', 
    elements_to_coordinate_systems=None, 
    table_key=None, 
    library_key=None, 
    coord_type=None, 
    n_neighs=6, 
    radius=None, 
    delaunay=False, 
    n_rings=1, 
    percentile=None, 
    transform=None, 
    set_diag=False, 
    key_added='spatial', 
    copy=False
)
```

Create a graph from spatial coordinates.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | `AnnData \| SpatialData` | Annotated data object. |
| `spatial_key` | `str` | Key in `anndata.AnnData.obsm` where spatial coordinates are stored. If adata is a `spatialdata.SpatialData`, the coordinates of the centroids will be stored in the adata with this key. |
| `elements_to_coordinate_systems` | `Optional[dict[str, str]]` | A dictionary mapping element names of the SpatialData object to coordinate systems. The elements can be either Shapes or Labels. For compatibility, the spatialdata table must annotate all regions keys. Must not be None if adata is a `spatialdata.SpatialData`. |
| `table_key` | `Optional[str]` | Key in `spatialdata.SpatialData.tables` where the spatialdata table is stored. Must not be None if adata is a `spatialdata.SpatialData`. |
| `library_key` | `Optional[str]` | If multiple library_id, column in `anndata.AnnData.obs` which stores mapping between library_id and obs. |
| `coord_type` | `Union[str, CoordType, None]` | Type of coordinate system:<br>- `'grid'` - grid coordinates.<br>- `'generic'` - generic coordinates.<br>- `None` - 'grid' if spatial_key is in `anndata.AnnData.uns` with n_neighs = 6 (Visium), otherwise use 'generic'. |
| `n_neighs` | `int` | Depending on the coord_type:<br>- `'grid'` - number of neighboring tiles.<br>- `'generic'` - number of neighborhoods for non-grid data. Only used when delaunay = False. |
| `radius` | `Union[float, tuple[float, float], None]` | Only available when coord_type = 'generic'. Depending on the type:<br>- `float` - compute the graph based on neighborhood radius.<br>- `tuple` - prune the final graph to only contain edges in interval [min(radius), max(radius)]. |
| `delaunay` | `bool` | Whether to compute the graph from Delaunay triangulation. Only used when coord_type = 'generic'. |
| `n_rings` | `int` | Number of rings of neighbors for grid data. Only used when coord_type = 'grid'. |
| `percentile` | `Optional[float]` | Percentile of the distances to use as threshold. Only used when coord_type = 'generic'. |
| `transform` | `Union[str, Transform, None]` | Type of adjacency matrix transform:<br>- `'spectral'` - spectral transformation of the adjacency matrix.<br>- `'cosine'` - cosine transformation of the adjacency matrix.<br>- `None` - no transformation of the adjacency matrix. |
| `set_diag` | `bool` | Whether to set the diagonal of the spatial connectivities to 1.0. |
| `key_added` | `str` | Key which controls where the results are saved if copy = False. |
| `copy` | `bool` | If True, return the result, otherwise save it to the adata object. |

### Returns

| Return Type | Description |
|-------------|-------------|
| `tuple[csr_matrix, csr_matrix] \| None` | If copy = True, returns a tuple with the spatial connectivities and distances matrices.<br>Otherwise, modifies the adata with the following keys:<br>- `anndata.AnnData.obsp['{key_added}_connectivities']` - the spatial connectivities.<br>- `anndata.AnnData.obsp['{key_added}_distances']` - the spatial distances.<br>- `anndata.AnnData.uns['{key_added}']` - dict containing parameters. |

---

## nhood_enrichment

```python
squidpy.gr.nhood_enrichment(
    adata, 
    cluster_key, 
    library_key=None, 
    connectivity_key=None, 
    n_perms=1000, 
    numba_parallel=False, 
    seed=None, 
    copy=False, 
    n_jobs=None, 
    backend='loky', 
    show_progress_bar=True
)
```

Compute neighborhood enrichment by permutation test.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | `AnnData \| SpatialData` | Annotated data object. |
| `cluster_key` | `str` | Key in `anndata.AnnData.obs` where clustering is stored. |
| `library_key` | `Optional[str]` | If multiple library_id, column in `anndata.AnnData.obs` which stores mapping between library_id and obs. |
| `connectivity_key` | `Optional[str]` | Key in `anndata.AnnData.obsp` where spatial connectivities are stored. Default is: `anndata.AnnData.obsp['spatial_connectivities']`. |
| `n_perms` | `int` | Number of permutations for the permutation test. |
| `numba_parallel` | `bool` | Whether to use numba.prange or not. If None, it is determined automatically. For small datasets or small number of interactions, it's recommended to set this to False. |
| `seed` | `Optional[int]` | Random seed for reproducibility. |
| `copy` | `bool` | If True, return the result, otherwise save it to the adata object. |
| `n_jobs` | `Optional[int]` | Number of parallel jobs. |
| `backend` | `str` | Parallelization backend to use. See joblib.Parallel for available options. |
| `show_progress_bar` | `bool` | Whether to show the progress bar or not. |

### Returns

| Return Type | Description |
|-------------|-------------|
| `tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]] \| None` | If copy = True, returns a tuple with the z-score and the enrichment count.<br>Otherwise, modifies the adata with the following keys:<br>- `anndata.AnnData.uns['{cluster_key}_nhood_enrichment']['zscore']` - the enrichment z-score.<br>- `anndata.AnnData.uns['{cluster_key}_nhood_enrichment']['count']` - the enrichment count. |

---

## spatial_autocorr

```python
squidpy.gr.spatial_autocorr(
    adata, 
    connectivity_key='spatial_connectivities', 
    genes=None, 
    mode='moran', 
    transformation=True, 
    n_perms=None, 
    two_tailed=False, 
    corr_method='fdr_bh', 
    attr='X', 
    layer=None, 
    seed=None, 
    use_raw=False, 
    copy=False, 
    n_jobs=None, 
    backend='loky', 
    show_progress_bar=True
)
```

Calculate Global Autocorrelation Statistic (Moran's I or Geary's C).

See [Rey and Anselin, 2010] for reference.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | `AnnData \| SpatialData` | Annotated data object. |
| `connectivity_key` | `str` | Key in `anndata.AnnData.obsp` where spatial connectivities are stored. Default is: `anndata.AnnData.obsp['spatial_connectivities']`. |
| `genes` | `Union[str, int, Sequence[str], Sequence[int], None]` | Depending on the attr:<br>- if attr = 'X', it corresponds to genes stored in `anndata.AnnData.var_names`. If None, it's computed `anndata.AnnData.var['highly_variable']`, if present. Otherwise, it's computed for all genes.<br>- if attr = 'obs', it corresponds to a list of columns in `anndata.AnnData.obs`. If None, use all numerical columns.<br>- if attr = 'obsm', it corresponds to indices in `anndata.AnnData.obsm['{layer}']`. If None, all indices are used. |
| `mode` | `Union[SpatialAutocorr, Literal['moran', 'geary']]` | Mode of score calculation:<br>- `'moran'` - Moran's I autocorrelation.<br>- `'geary'` - Geary's C autocorrelation. |
| `transformation` | `bool` | If True, weights in `anndata.AnnData.obsp['spatial_connectivities']` are row-normalized, advised for analytic p-value calculation. |
| `n_perms` | `Optional[int]` | Number of permutations for the permutation test. If None, only p-values under normality assumption are computed. |
| `two_tailed` | `bool` | If True, p-values are two-tailed, otherwise they are one-tailed. |
| `corr_method` | `str \| None` | Correction method for multiple testing. See `statsmodels.stats.multitest.multipletests()` for valid options. |
| `use_raw` | `bool` | Whether to access `anndata.AnnData.raw`. Only used when attr = 'X'. |
| `layer` | `Optional[str]` | Depending on attr: Layer in `anndata.AnnData.layers` to use. If None, use `anndata.AnnData.X`. |
| `attr` | `Literal['obs', 'X', 'obsm']` | Which attribute of AnnData to access. See genes parameter for more information. |
| `seed` | `Optional[int]` | Random seed for reproducibility. |
| `copy` | `bool` | If True, return the result, otherwise save it to the adata object. |
| `n_jobs` | `Optional[int]` | Number of parallel jobs. |
| `backend` | `str` | Parallelization backend to use. See `joblib.Parallel` for available options. |
| `show_progress_bar` | `bool` | Whether to show the progress bar or not. |

### Returns

| Return Type | Description |
|-------------|-------------|
| `DataFrame \| None` | If copy = True, returns a `pandas.DataFrame` with the following keys:<br>- 'I' or 'C' - Moran's I or Geary's C statistic.<br>- 'pval_norm' - p-value under normality assumption.<br>- 'var_norm' - variance of 'score' under normality assumption.<br>- '{p_val}_{corr_method}' - the corrected p-values if corr_method != None.<br><br>If n_perms != None, additionally returns the following columns:<br>- 'pval_z_sim' - p-value based on standard normal approximation from permutations.<br>- 'pval_sim' - p-value based on permutations.<br>- 'var_sim' - variance of 'score' from permutations.<br><br>Otherwise, modifies the adata with the following key:<br>- `anndata.AnnData.uns['moranI']` - the above mentioned dataframe, if mode = 'moran'.<br>- `anndata.AnnData.uns['gearyC']` - the above mentioned dataframe, if mode = 'geary'. |
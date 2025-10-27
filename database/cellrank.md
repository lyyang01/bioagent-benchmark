# CellRank
## The import form of the CellRank package
import cellrank
## Kernel which computes a transition matrix based on RNA velocity.
cellrank.kernels.VelocityKernel(adata, backward=False, attr='layers', xkey='Ms', vkey='velocity', **kwargs)[source]
### the detail description of parameters:
1. adata (AnnData) – Annotated data object.
2. backward (bool) – Direction of the process.
3. attr (Optional[Literal['layers', 'obsm']]) – Attribute of AnnData to read from.
4. xkey (Optional[str]) – Key in layers or obsm where expected gene expression counts are stored.
5. vkey (Optional[str]) – Key in layers or obsm where velocities are stored.
gene_subset – List of genes to be used to compute transition probabilities. If not specified, genes from adata.var['{vkey}_genes'] are used. This feature is only available when reading from anndata.AnnData.layers and will be ignored otherwise.
6. kwargs (Any) – Keyword arguments for the Kernel.
## Kernel which computes transition probabilities based on similarities among cells.
cellrank.kernels.ConnectivityKernel(adata, conn_key='connectivities', check_connectivity=False)
### the detail description of parameters:
1. adata (AnnData) – Annotated data object.
2. conn_key (str) – Key in obsp where connectivity matrix describing cell-cell similarity is stored.
3. check_connectivity (bool) – Check whether the underlying kNN graph is connected.
## Compute transition matrix.
compute_transition_matrix(*args, **kwargs)
### the detail description of parameters:
1. args (Any) – Positional arguments.
2. kwargs (Any) – Keyword arguments.
## Plot transition_matrix as a stream or a grid plot.
plot_projection(basis='umap', key_added=None, recompute=False, stream=True, connectivities=None, **kwargs)
### the detail description of parameters:
1. basis (str) – Key in obsm containing the basis.
2. key_added (Optional[str]) – If not None, save the result to adata.obsm['{key_added}']. Otherwise, save the result to 'T_fwd_{basis}' or 'T_bwd_{basis}', depending on the direction.
3. recompute (bool) – Whether to recompute the projection if it already exists.
4. stream (bool) – If True, use velocity_embedding_stream(). Otherwise, use velocity_embedding_grid().
5. connectivities (Optional[spmatrix]) – Connectivity matrix to use for projection. If None, use ones from the underlying kernel, is possible.
6. kwargs (Any) – Keyword argument for the above-mentioned plotting function.
## Plot random walks in an embedding.
plot_random_walks(n_sims=100, max_iter=0.25, seed=None, successive_hits=0, start_ixs=None, stop_ixs=None, basis='umap', cmap='gnuplot', linewidth=1.0, linealpha=0.3, ixs_legend_loc=None, n_jobs=None, backend='loky', show_progress_bar=True, figsize=None, dpi=None, save=None, **kwargs)
### the detail description of parameters:
n_sims (int) – Number of random walks to simulate.
max_iter (Union[int, float]) – Maximum number of steps of a random walk. If a float, it can be specified as a fraction of the number of cells.
seed (Optional[int]) – Random seed.
successive_hits (int) – Number of successive hits in the stop_ixs required to stop prematurely.
start_ixs (Union[Sequence[str], Mapping[str, Union[str, Sequence[str], tuple[float, float]]], None]) –
Cells from which to sample the starting points. If None, use all cells. Can be specified as:

dict - dictionary with 1 key in obs with values corresponding to either 1 or more clusters (if the column is categorical) or a tuple specifying  interval from which to select the indices.
Sequence - sequence of cell ids in obs_names.
For example {'dpt_pseudotime': [0, 0.1]} means that starting points for random walks will be sampled uniformly from cells whose pseudotime is in .

stop_ixs (Union[Sequence[str], Mapping[str, Union[str, Sequence[str], tuple[float, float]]], None]) –
Cells which when hit, the random walk is terminated. If None, terminate after max_iters. Can be specified as:

dict - dictionary with 1 key in obs with values corresponding to either 1 or more clusters (if the column is categorical) or a tuple specifying  interval from which to select the indices.
Sequence - sequence of cell ids in obs_names.
For example {'clusters': ['Alpha', 'Beta']} and successive_hits = 3 means that the random walk will stop prematurely after cells in the above specified clusters have been visited successively 3 times in a row.

basis (str) – Basis in obsm to use as an embedding.
cmap (Union[str, LinearSegmentedColormap]) – Colormap for the random walk lines.
linewidth (float) – Width of the random walk lines.
linealpha (float) – Alpha value of the random walk lines.
ixs_legend_loc (Optional[str]) – Legend location for the start/top indices.
show_progress_bar (bool) – Whether to show a progress bar. Disabling it may slightly improve performance.
n_jobs (Optional[int]) – Number of parallel jobs. If -1, use all available cores. If None or 1, the execution is sequential.
backend (str) – Which backend to use for parallelization. See Parallel for valid options.
figsize (Optional[tuple[float, float]]) – Size of the figure.
dpi (Optional[int]) – Dots per inch.
save (Union[Path, str, None]) – Filename where to save the plot.
kwargs (Any) – Keyword arguments for scatter().

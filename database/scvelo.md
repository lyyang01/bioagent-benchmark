# scVelo
## The import form of the scVelo package
import scvelo as scv
## Filter genes based on number of cells or counts
scvelo.pp.filter_genes(data, min_counts=None, min_cells=None, max_counts=None, max_cells=None, min_counts_u=None, min_cells_u=None, max_counts_u=None, max_cells_u=None, min_shared_counts=None, min_shared_cells=None, retain_genes=None, copy=False)
### the detail description of parameters:
data (AnnData, np.ndarray, sp.spmatrix) – The (annotated) data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.
min_counts (int, optional (default: None)) – Minimum number of counts required for a gene to pass filtering.
min_cells (int, optional (default: None)) – Minimum number of cells expressed required for a gene to pass filtering.
max_counts (int, optional (default: None)) – Maximum number of counts required for a gene to pass filtering.
max_cells (int, optional (default: None)) – Maximum number of cells expressed required for a gene to pass filtering.
min_counts_u (int, optional (default: None)) – Minimum number of unspliced counts required for a gene to pass filtering.
min_cells_u (int, optional (default: None)) – Minimum number of unspliced cells expressed required to pass filtering.
max_counts_u (int, optional (default: None)) – Maximum number of unspliced counts required for a gene to pass filtering.
max_cells_u (int, optional (default: None)) – Maximum number of unspliced cells expressed required to pass filtering.
min_shared_counts (int, optional (default: None)) – Minimum number of counts (both unspliced and spliced) required for a gene.
min_shared_cells (int, optional (default: None)) – Minimum number of cells required to be expressed (both unspliced and spliced).
retain_genes (list, optional (default: None)) – List of gene names to be retained independent of thresholds.
copy (bool, optional (default: False)) – Determines whether a copy is returned.
### return type
Filters the object and adds n_counts to adata.var.
## Computes moments for velocity estimation.
scvelo.pp.moments(data, n_neighbors=30, n_pcs=None, mode='connectivities', method='umap', use_rep=None, use_highly_variable=True, copy=False)
### the detail description of parameters:
data (AnnData) – Annotated data matrix.
n_neighbors (int (default: 30)) – Number of neighbors to use.
n_pcs (int (default: None)) – Number of principal components to use. If not specified, the full space is used of a pre-computed PCA, or 30 components are used when PCA is computed internally.
mode (‘connectivities’ or ‘distances’ (default: ‘connectivities’)) – Distance metric to use for moment computation.
method ({{‘umap’, ‘hnsw’, ‘sklearn’, None}} (default: ‘umap’)) – Method to compute neighbors, only differs in runtime. Connectivities are computed with adaptive kernel width as proposed in [Haghverdi et al., 2016].
use_rep (None, ‘X’ or any key for .obsm (default: None)) – Use the indicated representation. If None, the representation is chosen automatically: for .n_vars < 50, .X is used, otherwise ‘X_pca’ is used.
use_highly_variable (bool (default: True)) – Whether to use highly variable genes only, stored in .var[‘highly_variable’].
copy (bool (default: False)) – Return a copy instead of writing to adata.
### returns
Ms (.layers) – dense matrix with first order moments of spliced counts.
Mu (.layers) – dense matrix with first order moments of unspliced counts.
## Recovers the full splicing kinetics of specified genes
scvelo.tl.recover_dynamics(data, var_names='velocity_genes', n_top_genes=None, max_iter=10, assignment_mode='projection', t_max=None, fit_time=True, fit_scaling=True, fit_steady_states=True, fit_connected_states=None, fit_basal_transcription=None, use_raw=False, load_pars=None, return_model=None, plot_results=False, steady_state_prior=None, add_key='fit', copy=False, n_jobs=None, backend='loky', show_progress_bar=True, **kwargs)
### the detail description of parameters:
data (AnnData) – Annotated data matrix.
var_names (str, list of str (default: ‘velocity_genes’)) – Names of variables/genes to use for the fitting. If var_names=’velocity_genes’ but there is no column ‘velocity_genes’ in adata.var, velocity genes are estimated using the steady state model.
n_top_genes (int or None (default: None)) – Number of top velocity genes to use for the dynamical model.
max_iter (int (default: 10)) – Maximal iterations in the EM-Algorithm.
assignment_mode (str (default: projection)) – Determined how times are assigned to observations. If projection, observations are projected onto the model trajectory. Else uses an inverse approximating formula.
t_max (float, False or None (default: None)) – Total range for time assignments.
fit_scaling (bool or float or None (default: True)) – Whether to fit scaling between unspliced and spliced.
fit_time (bool or float or None (default: True)) – Whether to fit time or keep initially given time fixed.
fit_steady_states (bool or None (default: True)) – Whether to explicitly model and fit steady states (next to induction/repression)
fit_connected_states (bool or None (default: None)) – Restricts fitting to neighbors given by connectivities.
fit_basal_transcription (bool or None (default: None)) – Enables model to incorporate basal transcriptions.
use_raw (bool or None (default: None)) – if True, use .layers[‘sliced’], else use moments from .layers[‘Ms’]
load_pars (bool or None (default: None)) – Load parameters from past fits.
return_model (bool or None (default: True)) – Whether to return the model as :DynamicsRecovery: object.
plot_results (bool or None (default: False)) – Plot results after parameter inference.
steady_state_prior (list of bool or None (default: None)) – Mask for indices used for steady state regression.
add_key (str (default: ‘fit’)) – Key to add to parameter names, e.g. ‘fit_t’ for fitted time.
copy (bool (default: False)) – Return a copy instead of writing to adata.
n_jobs (int or None (default: None)) – Number of parallel jobs.
backend (str (default: “loky”)) – Backend used for multiprocessing. See joblib.Parallel for valid options.
show_progress_bar (bool) – Whether to show a progress bar.
### returns
data – Updated AnnData with inferred parameters added to .var if copy=True. The inferred parameters are the transcription rates fit_alpha, splicing rates fit_beta, degradation rates fit_gamma, switching times fit_t_, variance scaling factor for unspliced and spliced counts, model likelihoods fit_likelihood, and the scaling factor to align gene-wise latent times to a universal latent time fit_alignment_scaling.
## Estimates velocities in a gene-specific manner
scvelo.tl.velocity(data, vkey='velocity', mode='stochastic', fit_offset=False, fit_offset2=False, filter_genes=False, groups=None, groupby=None, groups_for_fit=None, constrain_ratio=None, use_raw=False, use_latent_time=None, perc=None, min_r2=0.01, min_likelihood=0.001, r2_adjusted=None, use_highly_variable=True, diff_kinetics=None, copy=False, **kwargs)
### the detail description of parameters:
data (AnnData) – Annotated data matrix.
vkey (str (default: ‘velocity’)) – Name under which to refer to the computed velocities for velocity_graph and velocity_embedding.
mode (‘deterministic’, ‘stochastic’ or ‘dynamical’ (default: ‘stochastic’)) – Whether to run the estimation using the steady-state/deterministic, stochastic or dynamical model of transcriptional dynamics. The dynamical model requires to run tl.recover_dynamics first.
fit_offset (bool (default: False)) – Whether to fit with offset for first order moment dynamics.
fit_offset2 (bool, (default: False)) – Whether to fit with offset for second order moment dynamics.
filter_genes (bool (default: True)) – Whether to remove genes that are not used for further velocity analysis.
groups (str, list (default: None)) – Subset of groups, e.g. [‘g1’, ‘g2’, ‘g3’], to which velocity analysis shall be restricted.
groupby (str, list or np.ndarray (default: None)) – Key of observations grouping to consider.
groups_for_fit (str, list or np.ndarray (default: None)) – Subset of groups, e.g. [‘g1’, ‘g2’, ‘g3’], to which steady-state fitting shall be restricted.
constrain_ratio (float or tuple of type float or None: (default: None)) – Bounds for the steady-state ratio.
use_raw (bool (default: False)) – Whether to use raw data for estimation.
use_latent_time (bool`or `None (default: None)) – Whether to use latent time as a regularization for velocity estimation.
perc (float (default: [5, 95])) – Percentile, e.g. 98, for extreme quantile fit.
min_r2 (float (default: 0.01)) – Minimum threshold for coefficient of determination
min_likelihood (float (default: None)) – Minimal likelihood for velocity genes to fit the model on.
r2_adjusted (bool (default: None)) – Whether to compute coefficient of determination on full data fit (adjusted) or extreme quantile fit (None)
use_highly_variable (bool (default: True)) – Whether to use highly variable genes only, stored in .var[‘highly_variable’].
copy (bool (default: False)) – Return a copy instead of writing to adata.
### returns
velocity (.layers) – velocity vectors for each individual cell
velocity_genes, velocity_beta, velocity_gamma, velocity_r2 (.var) – parameters
## Computes velocity graph based on cosine similarities
scvelo.tl.velocity_graph(data, vkey='velocity', xkey='Ms', tkey=None, basis=None, n_neighbors=None, n_recurse_neighbors=None, random_neighbors_at_max=None, sqrt_transform=None, variance_stabilization=None, gene_subset=None, compute_uncertainties=None, approx=None, mode_neighbors='distances', copy=False, n_jobs=None, backend='loky', show_progress_bar=True)
### the detail description of parameters:
data (AnnData) – Annotated data matrix.
vkey (str (default: ‘velocity’)) – Name of velocity estimates to be used.
xkey (str (default: ‘Ms’)) – Layer key to extract count data from.
tkey (str (default: None)) – Observation key to extract time data from.
basis (str (default: None)) – Basis / Embedding to use.
n_neighbors (int or None (default: None)) – Use fixed number of neighbors or do recursive neighbor search (if None).
n_recurse_neighbors (int (default: None)) – Number of recursions for neighbors search. Defaults to 2 if mode_neighbors is ‘distances’, and 1 if mode_neighbors is ‘connectivities’.
random_neighbors_at_max (int or None (default: None)) – If number of iterative neighbors for an individual cell is higher than this threshold, a random selection of such are chosen as reference neighbors.
sqrt_transform (bool (default: False)) – Whether to variance-transform the cell states changes and velocities before computing cosine similarities.
gene_subset (list of str, subset of adata.var_names or None`(default: `None)) – Subset of genes to compute velocity graph on exclusively.
compute_uncertainties (bool (default: None)) – Whether to compute uncertainties along with cosine correlation.
approx (bool or None (default: None)) – If True, first 30 pc’s are used instead of the full count matrix
mode_neighbors (‘str’ (default: ‘distances’)) – Determines the type of KNN graph used. Options are ‘distances’ or ‘connectivities’. The latter yields a symmetric graph.
copy (bool (default: False)) – Return a copy instead of writing to adata.
n_jobs (int or None (default: None)) – Number of parallel jobs.
backend (str (default: “loky”)) – Backend used for multiprocessing. See joblib.Parallel for valid options.
show_progress_bar (bool) – Whether to show a progress bar.
### returns
velocity_graph – sparse matrix with correlations of cell state transitions with velocities
## Stream plot of velocities on the embedding.
scvelo.pl.velocity_embedding_stream(adata, basis=None, vkey='velocity', density=2, smooth=None, min_mass=None, cutoff_perc=None, arrow_color=None, arrow_size=1, arrow_style='-|>', max_length=4, integration_direction='both', linewidth=None, n_neighbors=None, recompute=None, color=None, use_raw=None, layer=None, color_map=None, colorbar=True, palette=None, size=None, alpha=0.3, perc=None, X=None, V=None, X_grid=None, V_grid=None, sort_order=True, groups=None, components=None, legend_loc='on data', legend_fontsize=None, legend_fontweight=None, xlabel=None, ylabel=None, title=None, fontsize=None, figsize=None, dpi=None, frameon=None, show=None, save=None, ax=None, ncols=None, **kwargs)
### the detail description of parameters:
adata: AnnData. Annotated data matrix.
basis: str or list of str (default: None) Key for embedding. If not specified, use ‘umap’, ‘tsne’ or ‘pca’ (ordered by preference).
## Scatter plot along observations or variables axes.
scvelo.pl.scatter(adata=None, basis=None, x=None, y=None, vkey=None, color=None, use_raw=None, layer=None, color_map=None, colorbar=None, palette=None, size=None, alpha=None, linewidth=None, linecolor=None, perc=None, groups=None, sort_order=True, components=None, projection=None, legend_loc=None, legend_loc_lines=None, legend_fontsize=None, legend_fontweight=None, legend_fontoutline=None, legend_align_text=None, xlabel=None, ylabel=None, title=None, fontsize=None, figsize=None, xlim=None, ylim=None, add_density=None, add_assignments=None, add_linfit=None, add_polyfit=None, add_rug=None, add_text=None, add_text_pos=None, add_margin=None, add_outline=None, outline_width=None, outline_color=None, n_convolve=None, smooth=None, normalize_data=None, rescale_color=None, color_gradients=None, dpi=None, frameon=None, zorder=None, ncols=None, nrows=None, wspace=None, hspace=None, show=None, save=None, ax=None, aspect='auto', **kwargs)
## Computes a gene-shared latent time.
scvelo.tl.latent_time(data, vkey='velocity', min_likelihood=0.1, min_confidence=0.75, min_corr_diffusion=None, weight_diffusion=None, root_key=None, end_key=None, t_max=None, copy=False)
## Plot time series for genes as heatmap.
scvelo.pl.heatmap(adata, var_names, sortby='latent_time', layer='Ms', color_map='viridis', col_color=None, palette='viridis', n_convolve=30, standard_scale=0, sort=True, colorbar=None, col_cluster=False, row_cluster=False, context=None, font_scale=None, figsize=(8, 4), show=None, save=None, **kwargs)
## Get dataframe for a specified adata key.
scvelo.get_df(data, keys=None, layer=None, index=None, columns=None, sort_values=None, dropna='all', precision=None)



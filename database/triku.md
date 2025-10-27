# triku
## The import form of the triku package
import triku
## Call the triku method using python directly. This function expects an annData object or a csv / txt matrix of n_cells x n_genes. The function then returns an array / list of the selected genes.
triku.tl.triku(object_triku: Union[scanpy.AnnData, str], n_features: Union[None, int] = None, use_raw: bool = True, n_divisions: Union[None, int] = None, s: Union[int, float] = - 0.01, n_windows: int = 75, min_knn: int = 6, name: Union[str, None] = None, dist_conn: str = 'dist', distance_correction: str = 'median', verbose: Union[None, str] = 'warning')
### the detail description of parameters:
1. object_triku (scanpy.AnnData or pandas.DataFrame) – Object with count matrix. If pandas.DataFrame, rows are cells and columns are genes.
2. n_features (int) – Number of features to select. If None, the number is chosen automatically.
3. use_raw (bool) – If True, selects the adata.raw, if it exists. To set the .raw propety, set as: adata.raw = adata. This matrix is adjusted to select the genes and cells that appear in the current adata. E.g. if we are running triku with a subpopulation, triku will select the cells from adata.raw of that subpopulation. If certain genes have been removed after saving the raw, triku will not consider the removed genes.
4. n_divisions (int, None) – If the array of counts is not integer, number of bins in which each unit will be divided to account for that effect. For example, if n_divisions is 10, then 0.12 and 0.13 would be in the same bin, and 0.12 and 0.34 in two different bins. If n_divisions is 2, the two cases would be in the same bin. The higher the number of divisions the more precise the calculation of distances will be. It will be more time consuming, though. If n_divisions is None, we will adjust it automatically.
5. s (float) – Correction factor for automatic feature selection. Negative values imply a selction of more genes, and positive values imply a selection of fewer genes. We recommend values between -0.1 and 0.1.
6. n_windows (int) – Number of windows used for median subtraction of Wasserstein distance.
7. min_knn (int) – minimum number of expressed cells based on the knn to apply the convolution. If a gene has less than min_knn expressing cells, Wasserstein distance is set to 0, and the convolution is set as the knn expression.
8. name (str) – Name of the run. If None, stores results in “triku_X”. Else, stores it in “triku_X_{name}”.
9. dist_conn (str) – Uses adata.obsp[“distances”] or adata.obsp[“connectivities”] for knn array construction. From empirical analysis, “conn” shows slightly better results, but is slower.
10. distance_correction (str) – When correcting distances, uses median or mean of the distances for each bin. By default is “median”.
11. verbose (str ['debug', 'triku', 'info', 'warning', 'error', 'critical']) – Logger verbosity output.

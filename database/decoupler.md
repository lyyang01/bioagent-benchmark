# Decoupler
## The import form of the Decoupler package
import decoupler
## Swaps an AnnData X matrix with a given layer. Generates a new object by default.
decoupler.swap_layer(adata, layer_key, X_layer_key='X', inplace=False)
### the detail description of parameters:
1. adata: AnnData
Annotated data matrix.
3. layer_key: str
.layers key to place in .X.
3. X_layer_key: str, None
.layers key where to move and store the original .X. If None, the original .X is discarded.
4. inplace: bool
If False, return a copy. Otherwise, do operation inplace and return None.
### returns
layer: AnnData, None. If inplace=False, new AnnData object.
## Over Representation Analysis (ORA) with decoupler
decoupler.run_ora(mat, net, source='source', target='target', n_up=None, n_bottom=0, n_background=20000, min_n=5, seed=42, verbose=False, use_raw=True)
### the detail description of parameters:
1. mat: list, DataFrame or AnnData
List of [features, matrix], dataframe (samples x features) or an AnnData instance.
2. net: DataFrame
Network in long format.
3. source: str
Column name in net with source nodes.
4. target: str
Column name in net with target nodes.
5. n_up: int, None
Number of top ranked features to select as observed features. By default is the top 5% of positive features.
6. n_bottom: int
Number of bottom ranked features to select as observed features.
7. n_background: int
Integer indicating the background size.
8. min_n: int
Minimum of targets per source. If less, sources are removed.
9. seed: int
Random seed to use.
10. verbose: bool
Whether to show progress.
11. use_raw: bool
Use raw attribute of mat if present.
### returns
1. estimate: DataFrame
ORA scores, which are the -log(p-values). Stored in .obsm[‘ora_estimate’] if mat is AnnData.
2. pvals: DataFrame
Obtained p-values. Stored in .obsm[‘ora_pvals’] if mat is AnnData.
## From an AnnData object with source activities stored in .obsm, generates a new AnnData object with activities in X. This allows to reuse many scanpy processing and visualization functions.
decoupler.get_acts(adata, obsm_key, dtype=<class 'numpy.float32'>)
### the detail description of parameters:
1. adata: AnnData
Annotated data matrix with activities stored in .obsm.
2. obsm_key: str
.osbm key to extract.
3. dtype: type
Type of float used.
### returns
acts: AnnData. New AnnData object with activities in X.
## Rank sources for characterizing groups.
decoupler.rank_sources_groups(adata, groupby, reference='rest', method='t-test_overestim_var')
### the detail description of parameters:
1. adata: AnnData
AnnData obtained after running decoupler.get_acts.
2. groupby: str
The key of the observations grouping to consider.
3. reference: str, list
Reference group or list of reference groups to use as reference.
4. method: str
Statistical method to use for computing differences between groups. Avaliable methods include: {'wilcoxon', 't-test', 't-test_overestim_var'}.
### returns
results: DataFrame with changes in source activity score between groups.
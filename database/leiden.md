# leiden

## The `scanpy.tl.leiden` function is used to cluster cells into subgroups using the Leiden algorithm. It improves upon the Louvain algorithm and is commonly used in single-cell analysis.


## setup data function of leiden

sc.tl.leiden(adata, resolution=1, restrict_to=None, random_state=0, key_added='leiden', 
             adjacency=None, directed=None, use_weights=True, n_iterations=-1, 
             partition_type=None, neighbors_key=None, obsp=None, copy=False, 
             flavor='leidenalg', **clustering_args)

## Detailed Parameter Descriptions:
	1.	adata (AnnData) – The annotated data matrix containing cells and features.
	2.	resolution (float, default: 1) – Controls the granularity of clustering; higher values lead to more clusters.
	3.	random_state (int | RandomState | None, default: 0) – Sets the initialization of the clustering process.
	4.	restrict_to (tuple[str, Sequence[str]] | None, default: None) – Restrict clustering to specific categories within a sample annotation.
	5.	key_added (str, default: ‘leiden’) – The key under which cluster labels are stored in adata.obs.
	6.	adjacency (csr_matrix | csc_matrix | None, default: None) – The adjacency matrix of the graph, defaulting to neighbor connectivities.
	7.	directed (bool | None, default: None) – Whether to treat the graph as directed or undirected.
	8.	use_weights (bool, default: True) – If True, edge weights are used in the computation.
	9.	n_iterations (int, default: -1) – Number of Leiden clustering iterations; -1 runs until optimal clustering is reached.
	10.	partition_type (type[MutableVertexPartition] | None, default: None) – Type of partition to use (defaults to RBConfigurationVertexPartition).
	11.	neighbors_key (str | None, default: None) – Specifies the key for neighbor connectivities in adata.obsp.
	12.	obsp (str | None, default: None) – Specifies an alternative key in adata.obsp for the adjacency matrix.
	13.	copy (bool, default: False) – If True, returns a copy of adata; otherwise, modifies adata in place.
	14.	flavor (Literal[‘leidenalg’, ‘igraph’], default: ‘leidenalg’) – Specifies which package’s Leiden implementation to use.
	15.	clustering_args – Additional parameters passed to find_partition() or igraph.Graph.community_leiden().

Return Type:
	•	Returns None if copy=False, otherwise returns an AnnData object with clustering results.

Output Fields:
	•	adata.obs['leiden' | key_added] (pandas Series with categorical dtype):
Stores the cluster labels (e.g., ‘0’, ‘1’, etc.) for each cell.
	•	adata.uns['leiden' | key_added]['params'] (dictionary):
Stores the values for parameters such as resolution, random_state, and n_iterations.


# GraphST
## The import form of the GraphST package
from GraphST.utils import clustering
from GraphST import GraphST
## Sets up GraphST model
GraphST.GraphST(adata, device, random_seed)
### the detail description of parameters:
1. adata: AnnData
2. device: str, "cpu" or "cuda"
3. random_seed: int
## train GraphST model
model.train()
## clustering
clustering(adata, n_clusters, method)
### the detail description of parameters:
1. adata: AnnData
2. n_clusters: int, the number of principal components to use for clustering.
3. method: str, the clustering method to use. Options include 'louvain', 'leiden', and 'kmeans'.
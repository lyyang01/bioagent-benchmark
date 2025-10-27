# SpaGCN
## The import form of the SpaGCN package
import SpaGCN
## Sets up SpaGCN model
model = SpaGCN.SpaGCN()
### the detail description of parameters:
None (uses parameters defined during initialization)
## Calculate adjacency matrix
SpaGCN.calculate_adj_matrix(x, y, x_pixel, y_pixel, image, beta, alpha, histology)
### the detail description of parameters:
x: list, the x-coordinates of the cells.
y: list, the y-coordinates of the cells.
x_pixel: list, the x-coordinates of the cells in pixel space (optional).
y_pixel: list, the y-coordinates of the cells in pixel space (optional).
image: numpy array, the histology image (optional).
beta: float, the parameter controlling the spread of the Gaussian kernel (default: 49).
alpha: float, the parameter controlling the weight of the histology information (default: 1).
histology: bool, whether to use histology information in the adjacency matrix calculation (default: True).
## Prefilter genes
SpaGCN.prefilter_genes(adata, min_cells)
### the detail description of parameters:
adata: AnnData, the annotated data matrix containing spatial transcriptomics data.
min_cells: int, the minimum number of cells in which a gene must be expressed to be retained (default: 3).
Prefilter special genes
SpaGCN.prefilter_specialgenes(adata)
the detail description of parameters:
adata: AnnData, the annotated data matrix containing spatial transcriptomics data.
## Search for optimal l
SpaGCN.search_l(p, adj, start, end, tol, max_run)
### the detail description of parameters:
p: float, the target percentage of total expression contributed by neighborhoods.
adj: numpy array, the adjacency matrix.
start: float, the starting value for the search range (default: 0.01).
end: float, the ending value for the search range (default: 1000).
tol: float, the tolerance for the search (default: 0.01).
max_run: int, the maximum number of iterations for the search (default: 100).
## Search for optimal resolution
SpaGCN.search_res(adata, adj, l, n_clusters, start, step, tol, lr, max_epochs, r_seed, t_seed, n_seed)
### the detail description of parameters:
adata: AnnData, the annotated data matrix containing spatial transcriptomics data.
adj: numpy array, the adjacency matrix.
l: float, the value of the parameter l.
n_clusters: int, the desired number of clusters.
start: float, the starting value for the search range (default: 0.7).
step: float, the step size for the search (default: 0.1).
tol: float, the tolerance for the search (default: 5e-3).
lr: float, the learning rate for the clustering algorithm (default: 0.05).
max_epochs: int, the maximum number of epochs for the clustering algorithm (default: 20).
r_seed: int, the random seed (default: 100).
t_seed: int, the torch seed (default: 100).
n_seed: int, the numpy seed (default: 100).
## Train SpaGCN model
model.train(adata, adj, init_spa, init, res, tol, lr, max_epochs)
the detail description of parameters:
adata: AnnData, the annotated data matrix containing spatial transcriptomics data.
adj: numpy array, the adjacency matrix.
init_spa: bool, whether to initialize with spatial information (default: True).
init: str, the initialization method. Options include 'louvain' and 'kmeans' (default: 'louvain').
res: float, the resolution parameter for initial clustering.
tol: float, the tolerance for convergence (default: 5e-3).
lr: float, the learning rate for the clustering algorithm (default: 0.05).
max_epochs: int, the maximum number of epochs for the clustering algorithm (default: 200).
## Refine clustering results
SpaGCN.refine(sample_id, pred, dis, shape)
### the detail description of parameters:
sample_id: list, the list of sample IDs.
pred: list, the list of initial predictions.
dis: numpy array, the distance matrix.
shape: str, the shape of the spatial data. Options include 'hexagon' for Visium data and 'square' for ST data (default: 'hexagon').


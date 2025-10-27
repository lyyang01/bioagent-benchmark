# sctransform
## load sctransform R packages
library(sctransform)
## SCTransform
```python
# Perform sctransform-based normalization
Seurat SCTransform(object, ...)
```
### The detail description of parameters:
1. object: Seurat object
    - The Seurat object containing the raw UMI count matrix to be normalized.
2. assay: str (default: "RNA")
    - The name of the assay to pull the count data from. Default is 'RNA'.
3. new.assay.name: str (default: "SCT")
    - The name for the new assay containing the normalized data. Default is 'SCT'.
4. reference.SCT.model: SCTModel or None (default: None)
    - If not None, compute residuals for the object using the provided SCT model. Only supports log_umi as the latent variable.
5. do.correct.umi: bool (default: True)
    - Whether to place corrected UMI matrix in the assay counts slot. Default is True.
6. ncells: int (default: 5000)
    - Number of subsampling cells used to build the negative binomial regression model. Default is 5000.
7. residual.features: list of str or None (default: None)
    - Genes to calculate residual features for. If None, all genes will be used.
8. variable.features.n: int (default: 3000)
    - Number of features to use as variable features after ranking by residual variance. Default is 3000.
9. variable.features.rv.th: float (default: 1.3)
    - Residual variance threshold for selecting variable features. Default is 1.3.
10. vars.to.regress: list of str or None (default: None)
    - Variables to regress out in a second non-regularized linear regression. For example, percent.mito.
11. latent.data: matrix or None (default: None)
    - Extra data to regress out, should be cells x latent data.
12. do.scale: bool (default: False)
    - Whether to scale residuals to have unit variance. Default is False.
13. do.center: bool (default: True)
    - Whether to center residuals to have mean zero. Default is True.
14. clip.range: tuple (default: (-sqrt(n/30), sqrt(n/30)))
    - Range to clip the residuals to. Default is (-sqrt(n/30), sqrt(n/30)), where n is the number of cells.
15. vst.flavor: str (default: "v2")
    - Flavor of variance stabilizing transformation. Default is "v2".
16. conserve.memory: bool (default: False)
    - If True, the residual matrix for all genes is never created in full. Default is False.
17. return.only.var.genes: bool (default: True)
    - If True, the scale.data matrices in the output assay are subset to contain only the variable genes. Default is True.
18. seed.use: int or None (default: 1448145)
    - Random seed for reproducibility. Default is 1448145.
19. verbose: bool (default: True)
    - Whether to print messages and progress bars. Default is True.
## VlnPlot
```python
# Violin plot visualization
Seurat VlnPlot(object, features, group.by = NULL, split.by = NULL, idents = NULL, pt.size = 0, cols = NULL, y.max = NULL, y.min = NULL, x.label = NULL, y.label = NULL, log = FALSE, combine = TRUE, ncol = NULL, slot = "data", ...)
```
### The detail description of parameters:
1. object: Seurat object
    - The Seurat object containing the data to be plotted.
2. features: list of str
    - A list of features to plot. These can be gene names or any other feature names present in the Seurat object.
3. group.by: str or None (default: None)
    - A metadata column to group cells by for coloring. For example, "ident" or "celltype".
4. split.by: str or None (default: None)
    - A metadata column to split the plot by. For example, "ident" or "celltype".
5. idents: list of str or None (default: None)
    - A list of identity classes to include in the plot. If None, all identity classes are included.
6. pt.size: float (default: 0)
    - Size of the points on the plot.
7. cols: list of str or None (default: None)
    - Colors to use for the violin plots. If None, default colors will be used.
8. y.max: float or None (default: None)
    - Maximum y-axis value.
9. y.min: float or None (default: None)
    - Minimum y-axis value.
10. x.label: str or None (default: None)
    - Label for the x-axis.
11. y.label: str or None (default: None)
    - Label for the y-axis.
12. log: bool (default: False)
    - Whether to plot the y-axis on a log scale.
13. combine: bool (default: True)
    - Whether to combine multiple violin plots into a single plot. If False, a list of plots will be returned.
14. ncol: int or None (default: None)
    - Number of columns for the plot layout.
15. slot: str (default: "data")
    - Slot to pull expression data from. Default is "data".
16. ...: additional arguments
    - Additional arguments passed to other methods.
## Run a PCA dimensionality reduction.
Seurat.RunPCA(object, assay = "RNA", npcs = 50, verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to run PCA on.
2. assay: str (default: "RNA")
Name of Assay PCA is being run on.
3. npcs: int (default: 50)
Total Number of PCs to compute and store.
4. verbose: bool (default: True)
Print the top genes associated with high/low loadings for the PCs.
## Compute the k-param nearest neighbors for a given dataset.
Seurat.FindNeighbors(object, reduction = "pca", dims = 1:10, k.param = 20, verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to find neighbors for.
2. reduction: str (default: "pca")
Which dimensionality reduction to use.
3. dims: list of int (default: 1:10)
Which dimensions to use from the reduction.
4. k.param: int (default: 20)
Defines k for the k-nearest neighbor algorithm.
5. verbose: bool (default: True)
## Identify clusters of cells by a shared nearest neighbor (SNN) modularity optimization based clustering algorithm.
Seurat.FindClusters(object, resolution = 0.5, verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to find clusters for.
2. resolution: float (default: 0.5)
Value of the resolution parameter.
3. verbose: bool (default: True)
Print progress.
## Run UMAP dimensionality reduction.
Seurat.RunUMAP(object, reduction = "pca", dims = 1:10, n.neighbors = 30, verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to run UMAP on.
2. reduction: str (default: "pca")
Which dimensionality reduction to use.
3. dims: list of int (default: 1:10)
Which dimensions to use from the reduction.
4. n.neighbors: int (default: 30)
Number of neighbors to consider.
5. verbose: bool (default: True)
Print progress.
## Visualize features on a dimensional reduction plot.
Seurat.FeaturePlot(object, features, reduction = "umap", pt.size = None, cols = ["lightgrey", "blue"], verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to visualize features from.
2. features: list of str
Features to plot.
3. reduction: str (default: "umap")
Which dimensionality reduction to use.
4. pt.size: float (default: None)
Size of the points on the plot.
5. cols: list of str (default: ["lightgrey", "blue"])
Colors to form the gradient over.
6. verbose: bool (default: True)
Print progress.

# Seurat
## load Seurat R packages
library(Seurat)
## Collapse a multi-species expression matrix, keeping only one species of interest.
Seurat.CollapseSpeciesExpressionMatrix(object, prefix = "HUMAN_", controls = "MOUSE_", ncontrols = 100)
### the detail description of parameters:
1. object: Seurat object
The Seurat object containing the multi-species expression matrix.
2. prefix: str (default: "HUMAN_")
The prefix denoting rownames for the species of interest. Default is "HUMAN_". These rownames will have this prefix removed in the returned matrix.
3. controls: str (default: "MOUSE_")
The prefix denoting rownames for the species of 'negative control' cells. Default is "MOUSE_".
4. ncontrols: int (default: 100)
How many of the most highly expressed (average) negative control features (by default, 100 mouse genes), should be kept? All other rownames starting with controls are discarded.
## Create a Seurat object.
Seurat.CreateSeuratObject(counts, project = "SeuratProject", min.cells = 3, min.features = 200)
### the detail description of parameters:
1. counts: Matrix or DataFrame
The raw count matrix or DataFrame containing the gene expression data.
2. project: str (default: "SeuratProject")
The name of the project.
3. min.cells: int (default: 3)
Minimum number of cells expressing a feature to be included.
4. min.features: int (default: 200)
Minimum number of features detected in a cell to be included.
## Normalize the count data present in a given assay.
Seurat.NormalizeData(object, normalization.method = "LogNormalize", scale.factor = 10000)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to be normalized.
2. normalization.method: str (default: "LogNormalize")
Method for normalization. Options are "LogNormalize" or "CLR".
3. scale.factor: float (default: 10000)
Scale factor for normalization.
## Identify features that are outliers on a 'mean variability plot'.
Seurat.FindVariableFeatures(object, selection.method = "vst", nfeatures = 2000)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to identify variable features from.
2. selection.method: str (default: "vst")
How to choose top variable features. Options are "vst", "mean.var.plot", "dispersion", or "dispersion_norm".
3. nfeatures: int (default: 2000)
Number of features to select as top variable features.
## Scale and center the data.
Seurat.ScaleData(object, vars.to.regress = None, scale.max = 10)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to scale and center.
2. vars.to.regress: list of str (default: None)
Variables to regress out.
3. scale.max: float (default: 10)
Max value to return for scaled data.
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
## Finds markers (differentially expressed genes) for identity classes.
Seurat.FindMarkers(object, ident.1 = None, ident.2 = None, assay = "RNA", verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to find markers from.
2. ident.1: str (default: None)
Identity class to define markers for.
3. ident.2: str (default: None)
A second identity class for comparison.
4. assay: str (default: "RNA")
Assay to use in differential expression testing.
5. verbose: bool (default: True)
Print progress.
## Creates a scatter plot of two features across a set of single cells.
Seurat.FeatureScatter(object, feature1, feature2, pt.size = 1, cols = ["lightgrey", "blue"], verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to scatter plot features from.
2. feature1: str
First feature to plot.
3. feature2: str
Second feature to plot.
4. pt.size: float (default: 1)
Size of the points on the plot.
5. cols: list of str (default: ["lightgrey", "blue"])
Colors to form the gradient over.
6. verbose: bool (default: True)
Print progress.
## Visualize expression in a spatial context.
Seurat.SpatialFeaturePlot(object, features, image = None, crop = True, pt.size = 0.5, cols = ["lightgrey", "blue"], verbose = True)
### the detail description of parameters:
1. object: Seurat object
The Seurat object to visualize spatial features from.
2. features: list of str
Features to plot.
3. image: str (default: None)
Name of the image to use in the plot.
4. crop: bool (default: True)
Crop the plot to the area with cells.
5. pt.size: float (default: 0.5)
Size of the points on the plot.
6. cols: list of str (default: ["lightgrey", "blue"])
Colors to form the gradient over.
7. verbose: bool (default: True)
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

## SpatialDimPlot
```python
# Spatial dimension plot
Seurat SpatialDimPlot(object, fov = NULL, group.by = NULL, split.by = NULL, cols = NULL, pt.size = 0.5, shape.by = NULL, border.color = "white", border.size = NULL, na.value = "grey50", dark.background = TRUE, crop = FALSE, cells = NULL, combine = TRUE, coord.fixed = TRUE, flip_xy = TRUE)
```
### The detail description of parameters:
1. object: Seurat object
    - The Seurat object containing the spatial data.
2. fov: str or None (default: None)
    - Name of the field of view (FOV) to plot. If None, the first FOV will be used.
3. group.by: str or None (default: None)
    - Metadata column to group cells by for coloring.
4. split.by: str or None (default: None)
    - Metadata column to split the plot by.
5. cols: list of str or None (default: None)
    - Colors to use for the plot. If None, default colors will be used.
6. pt.size: float (default: 0.5)
    - Size of the points on the plot.
7. shape.by: str or None (default: None)
    - Metadata column to shape the points by.
8. border.color: str (default: "white")
    - Color of the cell segmentation borders.
9. border.size: float or None (default: None)
    - Size of the cell segmentation borders.
10. na.value: str (default: "grey50")
    - Color value for NA points.
11. dark.background: bool (default: True)
    - Whether to use a dark background for the plot.
12. crop: bool (default: False)
    - Whether to crop the plot to the area with cells.
13. cells: list of str or None (default: None)
    - List of cells to plot. If None, all cells will be plotted.
14. combine: bool (default: True)
    - Whether to combine multiple plots into a single plot.
15. coord.fixed: bool (default: True)
    - Whether to fix the cartesian coordinates.
16. flip_xy: bool (default: True)
    - Whether to flip the x and y coordinates.

## UpdateSeuratObject
```python
# Update Seurat object
Seurat UpdateSeuratObject(object)
```
### The detail description of parameters:
1. object: Seurat object
    - The Seurat object to be updated. This function updates the Seurat object to the latest version.

## FindTransferAnchors
```python
# Find transfer anchors
Seurat FindTransferAnchors(reference, query, normalization.method = "LogNormalize", recompute.residuals = TRUE, reference.assay = NULL, reference.neighbors = NULL, query.assay = NULL, reduction = "pcaproject", reference.reduction = NULL, project.query = FALSE, features = NULL, scale = TRUE, npcs = 30, l2.norm = TRUE, dims = 1:30, k.anchor = 5, k.filter = NA, k.score = 30, max.features = 200, nn.method = "annoy", n.trees = 50, eps = 0, approx.pca = TRUE, mapping.score.k = NULL, verbose = TRUE)
```
### The detail description of parameters:
1. reference: Seurat object
    - The reference Seurat object.
2. query: Seurat object
    - The query Seurat object.
3. normalization.method: str (default: "LogNormalize")
    - Normalization method used: "LogNormalize" or "SCT".
4. recompute.residuals: bool (default: True)
    - If using SCT normalization, compute query Pearson residuals using the reference SCT model parameters.
5. reference.assay: str or None (default: None)
    - Name of the assay to use from the reference.
6. reference.neighbors: str or None (default: None)
    - Name of the neighbor to use from the reference.
7. query.assay: str or None (default: None)
    - Name of the assay to use from the query.
8. reduction: str (default: "pcaproject")
    - Dimensional reduction to perform when finding anchors.
9. reference.reduction: str or None (default: None)
    - Name of the dimensional reduction to use from the reference.
10. project.query: bool (default: False)
    - Project the PCA from the query dataset onto the reference. Use only in rare cases.
11. features: list of str or None (default: None)
    - Features to use for dimensional reduction.
12. scale: bool (default: True)
    - Whether to scale the query data.
13. npcs: int (default: 30)
    - Number of PCs to compute on the reference.
14. l2.norm: bool (default: True)
    - Whether to perform L2 normalization on the cell embeddings.
15. dims: list of int (default: 1:30)
    - Dimensions to use from the reduction.
16. k.anchor: int (default: 5)
    - Number of neighbors to use when finding anchors.
17. k.filter: int or None (default: None)
    - Number of neighbors to use when filtering anchors.
18. k.score: int (default: 30)
    - Number of neighbors to use when scoring anchors.
19. max.features: int (default: 200)
    - Maximum number of features to use when specifying the neighborhood search space.
20. nn.method: str (default: "annoy")
    - Method for nearest neighbor finding.
21. n.trees: int (default: 50)
    - Number of trees for approximate nearest neighbor search.
22. eps: float (default: 0)
    - Error bound on the neighbor finding algorithm.
23. approx.pca: bool (default: True)
    - Whether to use truncated SVD for approximate PCA.
24. mapping.score.k: int or None (default: None)
    - Number of nearest neighbors to store for mapping score computation.
25. verbose: bool (default: True)
    - Whether to print progress messages.

## TransferData
```python
# Transfer data from reference to query
Seurat TransferData(anchorset, refdata, new.assay.name = NULL, reduction.model = NULL, verbose = TRUE)
```
### The detail description of parameters:
1. anchorset: AnchorSet object
    - The AnchorSet object generated by FindTransferAnchors.
2. refdata: Seurat object or str
    - Data to transfer from the reference. This can be a Seurat object or the name of a metadata field or assay in the reference object.
3. new.assay.name: str or None (default: None)
    - Name for the new assay containing the transferred data.
4. reduction.model: DimReduc object or None (default: None)
    - Dimensional reduction model to use for projecting the query data.
5. verbose: bool (default: True)
    - Whether to print progress messages.


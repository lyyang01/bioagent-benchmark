# scry
## load scry packages and other essential packages
library(scry)
library(DuoClustering2018)
library(SingleCellExperiment)
## Dimension Reduction with GLM-PCA
GLMPCA(sce, ncomp, assay = "counts", ...)
### the detail description of parameters:
1. sce: SingleCellExperiment object
The input SingleCellExperiment object containing the single-cell sequencing data. This object typically includes multiple assays (e.g., raw counts, log-normalized data) and metadata about the cells and genes.
2. ncomp: integer
The number of principal components to retain. This parameter controls the dimensionality reduction aspect of the GLMPCA. For example, setting ncomp = 2 will retain the first two principal components.
3. assay: character (default: "counts")
The name of the assay to use from the SingleCellExperiment object. Common assays include "counts" (raw counts), "logcounts" (log-normalized counts), or other custom assays added to the object. The default value is "counts", which is typically the raw count data.
4. ...: additional arguments
Additional arguments that can be passed to the underlying GLM or PCA functions. These might include parameters for the GLM fitting process (e.g., family, link function) or for the PCA implementation (e.g., scaling, centering).

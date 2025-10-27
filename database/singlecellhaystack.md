# singleCellHaystack
## load singleCellHaystack R package
library(singleCellHaystack)
## The main Haystack function
haystack(x, ...)
### the detail description of parameters:
1. **x**
   - A matrix or other object from which coordinates of cells can be extracted.
2. **...**
   - Further parameters passed down to methods.
3. **expression**
   - A matrix with expression data of genes (rows) in cells (columns).
4. **weights.advanced.Q**
   - If NULL, naive sampling is used. If a vector is given (of length = no. of cells), sampling is done according to the values in the vector.
5. **dir.randomization**
   - If NULL, no output is made about the random sampling step. If not NULL, files related to the randomizations are printed to this directory.
6. **scale**
   - Logical (default = TRUE) indicating whether input coordinates in x should be scaled to mean 0 and standard deviation 1.
7. **grid.points**
   - An integer specifying the number of centers (gridpoints) to be used for estimating the density distributions of cells. Default is set to 100.
8. **grid.method**
   - The method to decide grid points for estimating the density in the high-dimensional space. Should be "centroid" (default) or "seeding".
9. **coord**
   - Name of coordinates slot for specific methods.
10. **assay**
    - Name of assay data for Seurat method.
11. **slot**
    - Name of slot for assay data for Seurat method.
12. **dims**
    - Dimensions from coord to use. By default, all.
13. **cutoff**
    - Cutoff for detection.
14. **method**
    - Choose between highD (default) and 2D haystack.
## Shows the results of the 'haystack' analysis in various ways, sorted by significance. 
show_result_haystack(
  res.haystack,
  n = NULL,
  p.value.threshold = NULL,
  gene = NULL
)
### the detail description of parameters:
1. res.haystack
A 'haystack' result object.
2. n
If defined, the top "n" significant genes will be returned. Default: NA, which shows all results.
3. p.value.threshold
If defined, genes passing this p-value threshold will be returned.
4. gene
If defined, the results of this (these) gene(s) will be returned.
## Function for hierarchical clustering of genes according to their expression distribution in 2D or multi-dimensional space
hclust_haystack(
  x,
  expression,
  grid.coordinates,
  hclust.method = "ward.D",
  cor.method = "spearman",
  ...
)
### the detail description of parameters:
1. x
a matrix or other object from which coordinates of cells can be extracted.
2. expression
expression matrix.
3. grid.coordinates
coordinates of the grid points.
4. hclust.method
method used with hclust.
5. cor.method
method used with cor.
...
further parameters passed down to methods.


# scPNMF
## The import form of the scPNMF package
```R
library(scPNMF)
```
## Function: PNMFfun
```R
scPNMF::PNMFfun(X, K, method = "EucDist", tol = 1e-4, maxIter = 1000, verboseN = TRUE)
```
### The detailed description of parameters:
1. **X**: matrix  
   - The input data matrix, typically a gene expression matrix with genes as rows and cells as columns.
2. **K**: integer  
   - The number of latent factors (or components) to be used in the factorization. This parameter controls the dimensionality of the reduced space.
3. **method**: character (default: "EucDist")  
   - The distance metric used in the optimization process. The default value is "EucDist" for Euclidean distance.
4. **tol**: numeric (default: 1e-4)  
   - The tolerance for convergence. The algorithm stops if the change in the objective function is less than this value.
5. **maxIter**: integer (default: 1000)  
   - The maximum number of iterations for the optimization algorithm.
6. **verboseN**: logical (default: TRUE)  
   - Whether to print progress information during the optimization process.

## Function: basisAnnotate
```R
scPNMF::basisAnnotate(W, dim_use = 1:5, id_type = "ENSEMBL", return_fig = TRUE)
```
### The detailed description of parameters:
1. **W**: matrix  
   - The basis matrix obtained from the scPNMF factorization.
2. **dim_use**: integer vector (default: 1:5)  
   - The dimensions (columns) of the basis matrix to use for annotation.
3. **id_type**: character (default: "ENSEMBL")  
   - The type of gene identifiers used in the basis matrix (e.g., "ENSEMBL", "SYMBOL").
4. **return_fig**: logical (default: TRUE)  
   - Whether to return a figure summarizing the annotation results.

## Function: basisTest
```R
scPNMF::basisTest(S, X, return_fig = TRUE, ncol = 5, mc.cores = 1)
```
### The detailed description of parameters:
1. **S**: matrix  
   - The sparse encoding matrix obtained from the scPNMF factorization.
2. **X**: matrix  
   - The original input data matrix used for testing the basis genes.
3. **return_fig**: logical (default: TRUE)  
   - Whether to return a figure summarizing the test results.
4. **ncol**: integer (default: 5)  
   - The number of columns to display in the output figure.
5. **mc.cores**: integer (default: 1)  
   - The number of cores to use for parallel processing.
## Function: basisSelect
```R
scPNMF::basisSelect(W, S, X, toTest = TRUE, toAnnotate = FALSE, mc.cores = 1)
```
### The detailed description of parameters:
1. **W**: matrix  
   - The basis matrix obtained from the scPNMF factorization.
2. **S**: matrix  
   - The sparse encoding matrix obtained from the scPNMF factorization.
3. **X**: matrix  
   - The original input data matrix used for selecting the basis genes.
4. **toTest**: logical (default: TRUE)  
   - Whether to perform testing of the basis genes.
5. **toAnnotate**: logical (default: FALSE)  
   - Whether to annotate the selected basis genes.
6. **mc.cores**: integer (default: 1)  
   - The number of cores to use for parallel processing.
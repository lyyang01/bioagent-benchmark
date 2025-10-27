# Harmony
## The import form of the Harmony package
import scanpy.external
## Integration with harmony
scanpy.external.pp.harmony_integrate(adata, key, *, basis='X_pca', adjusted_basis='X_pca_harmony', **kwargs)
### the detail description of parameters:
1. adata: AnnData
The annotated data matrix that contains the single-cell data to be integrated.
2. key: str or Sequence[str]
The name(s) of the column(s) in adata.obs that differentiate among experiments or batches. For example, if you have a column named 'batch' in adata.obs that indicates the batch of each cell, you would pass 'batch' as the key. To integrate over multiple covariates, you can pass a list of column names.
3. basis: str (default: 'X_pca')
The name of the field in adata.obsm where the PCA table is stored. This is typically the output of sc.pp.pca(). The default value is 'X_pca', which is the standard PCA result field in Scanpy.
4. adjusted_basis: str (default: 'X_pca_harmony')
The name of the field in adata.obsm where the adjusted PCA table will be stored after running Harmony integration. The default value is 'X_pca_harmony', and this field will contain the batch-corrected principal components.
5. kwargs:
Additional keyword arguments that will be passed to harmonypy.run_harmony(). These can include parameters specific to the Harmony algorithm, such as theta (the weight for the batch correction) or sigma (the dispersion parameter)

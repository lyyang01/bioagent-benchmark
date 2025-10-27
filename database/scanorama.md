# Scanorama
## The import form of the Scanorama package
import scanorama
## Integration with correct_scanpy (It provides more parameter options and is more suitable for users who need fine-grained control over the integration process)
scanorama.correct_scanpy(adatas, return_dimred, dimred, hvg, n_top_genes, batch_key)
### the detail description of parameters:
1. adatas: A list of AnnData objects, where each AnnData object represents a dataset to be integrated.
2. return_dimred: Boolean, optional (default: False). If set to True, the function returns the dimensionality-reduced embeddings.
3. dimred: Integer, optional (default: 100). Specifies the number of dimensions for the reduced embeddings.
4. hvg: Boolean, optional (default: False). If set to True, highly variable genes are selected for each dataset before integration.
5. n_top_genes: Integer, optional. When hvg=True, this specifies the number of top highly variable genes to select.
7. batch_key: String, optional. Specifies the key in adata.obs where the batch information will be stored after integration.
### return value:
Returns a list of AnnData objects, each corresponding to the integrated result of the input datasets. If return_dimred=True, each AnnData object will have an embedding matrix stored in adata.obsm['X_scanorama']
## Integration with integration_scanpy (It has fewer parameters and is easier to use, making it suitable for quickly integrating data)
scanorama.integrate_scanpy(adatas, dimred, hvg, n_top_genes, batch_key)
### the detail description of parameters:
1. adatas: A list of AnnData objects, where each object represents a dataset to be integrated.
2. dimred: Integer, specifies the number of dimensions for the reduced embeddings (default: 100).
3. hvg: Boolean, optional (default: False). If set to True, highly variable genes are selected for each dataset before integration.
4. n_top_genes: Integer, optional. When hvg=True, this specifies the number of top highly variable genes to select.
5. batch_key: String, optional. Specifies the key in adata.obs where the batch information will be stored after integration.
### return value:
Returns a single AnnData object containing the integrated results of all input datasets. The dimensionality-reduced embeddings are stored in adata.obsm['X_scanorama']


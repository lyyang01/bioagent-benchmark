# CellAssign
## Marker-based automated cell annotation with CellAssign
## The import form of the CellAssign package
from scvi.external import CellAssign
## setup data function of CellAssign
CellAssign.setup_anndata(adata, size_factor_key, batch_key=None, categorical_covariate_keys=None, continuous_covariate_keys=None, layer=None, **kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object. Rows represent cells, columns represent features.
2. size_factor_key (str) – key in adata.obs with continuous valued size factors.
3. batch_key (str | None (default: None)) – key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
4. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
5. categorical_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to categorical data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
6. continuous_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to continuous data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
## setup model function of CellAssign
CellAssign(adata, cell_type_markers, **model_kwargs)
### the detail description of parameters:
1. adata (AnnData) – single-cell AnnData object that has been registered via setup_anndata(). The object should be subset to contain the same genes as the cell type marker dataframe.
2. cell_type_markers (DataFrame) – Binary marker gene DataFrame of genes by cell types. Gene names corresponding to adata.var_names should be in DataFrame index, and cell type labels should be the columns.
3. **model_kwargs – Keyword args for CellAssignModule
## how to train the CellAssign model
CellAssign.train(max_epochs=400, lr=0.003, accelerator='auto', devices='auto', train_size=0.9, validation_size=None, shuffle_set_split=True, batch_size=1024, datasplitter_kwargs=None, plan_kwargs=None, early_stopping=True, early_stopping_patience=15, early_stopping_min_delta=0.0, **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 400)) – Number of epochs to train for
2. lr (float (default: 0.003)) – Learning rate for optimization.
3. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
4. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
5. train_size (float (default: 0.9)) – Size of training set in the range [0.0, 1.0].
6. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
7. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
8. batch_size (int (default: 1024)) – Minibatch size to use during training.
9. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
10. plan_kwargs (dict | None (default: None)) – Keyword args for TrainingPlan.
11. early_stopping (bool (default: True)) – Adds callback for early stopping on validation_loss
12. early_stopping_patience (int (default: 15)) – Number of times early stopping metric can not improve over early_stopping_min_delta
13. early_stopping_min_delta (float (default: 0.0)) – Threshold for counting an epoch torwards patience train() will overwrite values present in 
14. plan_kwargs, when appropriate.
15. **kwargs – Other keyword args for Trainer.
## how to get results from trained model
CellAssign.predict()


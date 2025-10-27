# scVI
## cell integration with scVI
## The import form of the scvi package
import scvi
## setup data function of scvi
scvi.model.SCVI.setup_anndata(adata, layer=None, batch_key=None, labels_key=None, size_factor_key=None, categorical_covariate_keys=None, continuous_covariate_keys=None, **kwargs)
### the detail description of parameters:
1. adata (AnnData object, required): This is the input AnnData object that contains single-cell data. It typically contains a matrix of gene expression counts (adata.X) along with metadata about cells (adata.obs) and genes (adata.var).
2. batch_key (str, optional): This specifies the key in adata.obs that corresponds to the batch or experimental condition of each cell. It is useful for handling batch effects, which occur when cells from different experiments have variations in their expression profiles. If not provided, no batch information is used.
3. labels_key (str, optional): This specifies the key in adata.obs that corresponds to cell labels (i.e., cluster annotations or cell types). These labels are used for supervised learning tasks like classification. If not provided, the model will assume that no labels are used for training.
4. size_factor_key (str, optional): This specifies the key in adata.obs that corresponds to precomputed size factors (normalization factors) for each cell. Size factors are used to adjust for differences in sequencing depth between cells. If not provided, the model will compute size factors internally (using library size normalization).
5. layer (str, optional): This parameter specifies which layer of the AnnData object should be used for the count data. In AnnData, layers are alternative representations of the data (e.g., raw counts, normalized counts). If not provided, the method will use adata.X.
6. kwargs (dict, optional): Additional keyword arguments that are passed to the method for flexibility. These may include advanced options for customizing the setup process.
7. categorical_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to categorical data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
8. continuous_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to continuous data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
## setup model function of scvi
scvi.model.SCVI(adata=None, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, dispersion='gene', gene_likelihood='zinb', latent_distribution='normal', **kwargs)
### the detail description of parameters:
1. adata (AnnData | None (default: None)) AnnData object that has been registered via setup_anndata(). If None, then the underlying module will not be initialized until training, and a LightningDataModule must be passed in during training (EXPERIMENTAL).
2. n_hidden (int (default: 128)) Number of nodes per hidden layer.
3. n_latent (int (default: 10)) Dimensionality of the latent space.
4. n_layers (int (default: 1)) Number of hidden layers used for encoder and decoder NNs.
5. dropout_rate (float (default: 0.1)) Dropout rate for neural networks.
6. dispersion (Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] (default: 'gene'))
    One of the following:
    'gene' - dispersion parameter of NB is constant per gene across cells
    'gene-batch' - dispersion can differ between different batches
    'gene-label' - dispersion can differ between different labels
    'gene-cell' - dispersion can differ for every gene in every cell
7. gene_likelihood (Literal['zinb', 'nb', 'poisson'] (default: 'zinb')) �C
    One of:
    'nb' - Negative binomial distribution
    'zinb' - Zero-inflated negative binomial distribution
    'poisson' - Poisson distribution
8. latent_distribution (Literal['normal', 'ln'] (default: 'normal')) �C
    One of:
    'normal' - Normal distribution
    'ln' - Logistic normal distribution (Normal(0, I) transformed by softmax)
9. **kwargs C Additional keyword arguments for VAE.
## how to train the scvi model
model.train(max_epochs=None, accelerator='auto', devices='auto', train_size=0.9, validation_size=None, shuffle_set_split=True, load_sparse_tensor=False, batch_size=128, early_stopping=False, datasplitter_kwargs=None, plan_kwargs=None, data_module=None, **trainer_kwargs)
### the detail description of parameters:
1. max_epochs (int | None (default: None)) – The maximum number of epochs to train the model. The actual number of epochs may be less if early stopping is enabled. If None, defaults to a heuristic based on get_max_epochs_heuristic(). Must be passed in if data_module is passed in, and it does not have an n_obs attribute.
2. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
3. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
4. train_size (float (default: 0.9)) – Size of training set in the range [0.0, 1.0]. Passed into DataSplitter. Not used if data_module is passed in.
5. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set. Passed into DataSplitter. Not used if data_module is passed in.
6. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages. Passed into DataSplitter. Not used if data_module is passed in.
7. load_sparse_tensor (bool (default: False)) – EXPERIMENTAL If True, loads data with sparse CSR or CSC layout as a Tensor with the same layout. Can lead to speedups in data transfers to GPUs, depending on the sparsity of the data. Passed into DataSplitter. Not used if data_module is passed in.
8. batch_size (Tunable_[int] (default: 128)) – Minibatch size to use during training. Passed into DataSplitter. Not used if data_module is passed in.
9. early_stopping (bool (default: False)) – Perform early stopping. Additional arguments can be passed in through **kwargs. See Trainer for further options.
10. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter. Values in this argument can be overwritten by arguments directly passed into this method, when appropriate. Not used if data_module is passed in.
11. plan_kwargs (dict | None (default: None)) – Additional keyword arguments passed into TrainingPlan. Values in this argument can be overwritten by arguments directly passed into this method, when appropriate.
12. data_module (LightningDataModule | None (default: None)) – EXPERIMENTAL A LightningDataModule instance to use for training in place of the default DataSplitter. Can only be passed in if the model was not initialized with AnnData.
13. **kwargs – Additional keyword arguments passed into Trainer.
## how to get results from trained model
model.get_latent_representation()
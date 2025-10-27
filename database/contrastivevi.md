# contrastiveVI
## Isolating perturbation-induced variations with contrastiveVI
## The import form of the ContrastiveVI package
from scvi.external import ContrastiveVI
## setup data function of ContrastiveVI
ContrastiveVI.setup_anndata(adata, layer=None, batch_key=None, labels_key=None, size_factor_key=None, categorical_covariate_keys=None, continuous_covariate_keys=None, **kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object. Rows represent cells, columns represent features.
2. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
3. batch_key (str | None (default: None)) – key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
4. labels_key (str | None (default: None)) – key in adata.obs for label information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_labels’]. If None, assigns the same label to all the data.
5. size_factor_key (str | None (default: None)) – key in adata.obs for size factor information. Instead of using library size as a size factor, the provided size factor column will be used as offset in the mean of the likelihood. Assumed to be on linear scale.
6. categorical_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to categorical data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
7. continuous_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to continuous data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
## setup model function of ContrastiveVI
ContrastiveVI(adata, n_hidden=128, n_background_latent=10, n_salient_latent=10, n_layers=1, dropout_rate=0.1, use_observed_lib_size=True, wasserstein_penalty=0)
### the detail description of parameters:
1. adata (AnnData) – AnnData object that has been registered via setup_anndata().
2. n_hidden (int (default: 128)) – Number of nodes per hidden layer.
3. n_background_latent (int (default: 10)) – Dimensionality of the background shared latent space.
4. n_salient_latent (int (default: 10)) – Dimensionality of the salient latent space.
5. n_layers (int (default: 1)) – Number of hidden layers used for encoder and decoder NNs.
6. dropout_rate (float (default: 0.1)) – Dropout rate for neural networks.
7. use_observed_lib_size (bool (default: True)) – Use observed library size for RNA as scaling factor in mean of conditional distribution.
8. wasserstein_penalty (float (default: 0)) – Weight of the Wasserstein distance loss that further discourages background shared variations from leaking into the salient latent space.
## how to train the ContrastiveVI model
ContrastiveVI.train(background_indices, target_indices, max_epochs=None, accelerator='auto', devices='auto', train_size=0.9, validation_size=None, shuffle_set_split=True, load_sparse_tensor=False, batch_size=128, early_stopping=False, datasplitter_kwargs=None, plan_kwargs=None, **trainer_kwargs)
### the detail description of parameters:
1. max_epochs (int | None (default: None)) – Number of passes through the dataset. If None, defaults to np.min([round((20000 / n_cells) * 400), 400])
2. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
3. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
4. train_size (float (default: 0.9)) – Size of training set in the range [0.0, 1.0].
5. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
6. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
7. load_sparse_tensor (bool (default: False)) – EXPERIMENTAL If True, loads data with sparse CSR or CSC layout as a Tensor with the same layout. Can lead to speedups in data transfers to GPUs, depending on the sparsity of the data.
8. batch_size (Tunable_[int] (default: 128)) – Minibatch size to use during training.
9. early_stopping (bool (default: False)) – Perform early stopping. Additional arguments can be passed in **kwargs. See Trainer for further options.
10. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into ContrastiveDataSplitter.
11. plan_kwargs (dict | None (default: None)) – Keyword args for TrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
12. **trainer_kwargs – Other keyword args for Trainer.
## how to get results from trained model
ContrastiveVI.get_latent_representation()
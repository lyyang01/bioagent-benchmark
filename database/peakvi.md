# PeakVI
## chromatin accessibility analysis with PeakVI
## The import form of the peakVI package
import scvi
## setup data function of peakVI
scvi.model.PEAKVI.setup_anndata(adata, batch_key=None, labels_key=None, categorical_covariate_keys=None, continuous_covariate_keys=None, layer=None, **kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object. Rows represent cells, columns represent features.
2. batch_key (str | None (default: None)) – key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
3. labels_key (str | None (default: None)) – key in adata.obs for label information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_labels’]. If None, assigns the same label to all the data.
4. categorical_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to categorical data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
5. continuous_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to continuous data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
6. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
## setup model function of peakVI
scvi.model.PEAKVI(adata, n_hidden=None, n_latent=None, n_layers_encoder=2, n_layers_decoder=2, dropout_rate=0.1, model_depth=True, region_factors=True, use_batch_norm='none', use_layer_norm='both', latent_distribution='normal', deeply_inject_covariates=False, encode_covariates=False, **model_kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object that has been registered via setup_anndata().
2. n_hidden (int | None (default: None)) – Number of nodes per hidden layer. If None, defaults to square root of number of regions.
3. n_latent (int | None (default: None)) – Dimensionality of the latent space. If None, defaults to square root of n_hidden.
4. n_layers_encoder (int (default: 2)) – Number of hidden layers used for encoder NN.
5. n_layers_decoder (int (default: 2)) – Number of hidden layers used for decoder NN.
6. dropout_rate (float (default: 0.1)) – Dropout rate for neural networks
model_depth (bool (default: True)) – Model sequencing depth / library size (default: True)
7. region_factors (bool (default: True)) – Include region-specific factors in the model (default: True)
latent_distribution (Literal['normal', 'ln'] (default: 'normal')) –
    One of
        'normal' - Normal distribution (Default)
        'ln' - Logistic normal distribution (Normal(0, I) transformed by softmax)
8. deeply_inject_covariates (bool (default: False)) – Whether to deeply inject covariates into all layers of the decoder. If False (default), covariates will only be included in the input layer.
9. **model_kwargs – Keyword args for PEAKVAE
## how to train the peakVI model
scvi.model.PEAKVI.train(max_epochs=500, lr=0.0001, accelerator='auto', devices='auto', train_size=0.9, validation_size=None, shuffle_set_split=True, batch_size=128, weight_decay=0.001, eps=1e-08, early_stopping=True, early_stopping_patience=50, save_best=True, check_val_every_n_epoch=None, n_steps_kl_warmup=None, n_epochs_kl_warmup=50, datasplitter_kwargs=None, plan_kwargs=None, **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 500)) – Number of passes through the dataset.
lr (float (default: 0.0001)) – Learning rate for optimization.
2. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
3. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
4. train_size (float (default: 0.9)) – Size of training set in the range [0.0, 1.0].
5. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
6. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
7. batch_size (int (default: 128)) – Minibatch size to use during training.
weight_decay (float (default: 0.001)) – weight decay regularization term for optimization
8. eps (float (default: 1e-08)) – Optimizer eps
9. early_stopping (bool (default: True)) – Whether to perform early stopping with respect to the validation set.
10. early_stopping_patience (int (default: 50)) – How many epochs to wait for improvement before early stopping
11. save_best (bool (default: True)) – DEPRECATED Save the best model state with respect to the validation loss (default), or use the final state in the training procedure
12. check_val_every_n_epoch (int | None (default: None)) – Check val every n train epochs. By default, val is not checked, unless early_stopping is True. If so, val is checked every epoch.
13. n_steps_kl_warmup (int | None (default: None)) – Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1. Only activated when n_epochs_kl_warmup is set to None. If None, defaults to floor(0.75 * adata.n_obs).
14. n_epochs_kl_warmup (int | None (default: 50)) – Number of epochs to scale weight on KL divergences from 0 to 1. Overrides n_steps_kl_warmup when both are not None.
15. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
16. plan_kwargs (dict | None (default: None)) – Keyword args for TrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
17. **kwargs – Other keyword args for Trainer
## how to get results from trained model
model.get_latent_representation()
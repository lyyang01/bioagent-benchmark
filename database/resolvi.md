# ResolVI
## The import form of the ResolVI package
from scvi.external import RESOLVI
## setup data function of ResolVI
RESOLVI.setup_anndata(adata, layer=None, batch_key=None, labels_key=None, categorical_covariate_keys=None, prepare_data=True, prepare_data_kwargs=None, unlabeled_category='unknown', **kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object. Rows represent cells, columns represent features.
layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
2. batch_key (str | None (default: None)) – key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
3. labels_key (str | None (default: None)) – key in adata.obs for label information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_labels’]. If None, assigns the same label to all the data.
4. categorical_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to categorical data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
5. prepare_data (bool | None (default: True)) – If True, prepares AnnData for training. Computes spatial neighbors and distances.
6. prepare_data_kwargs (dict (default: None)) – Keyword args for scvi.external.RESOLVI._prepare_data()
7. unlabeled_category (str (default: 'unknown')) – value in adata.obs[labels_key] that indicates unlabeled observations.
## setup model function of ResolVI
RESOLVI(adata, n_hidden=32, n_hidden_encoder=128, n_latent=10, n_layers=2, dropout_rate=0.05, dispersion='gene', gene_likelihood='nb', background_ratio=None, median_distance=None, semisupervised=False, mixture_k=50, downsample_counts=True, **model_kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object that has been registered via setup_anndata().
2. n_hidden (int (default: 32)) – Number of nodes per hidden layer.
3. n_latent (int (default: 10)) – Dimensionality of the latent space.
4. n_layers (int (default: 2)) – Number of hidden layers used for encoder and decoder NNs.
5. dropout_rate (float (default: 0.05)) – Dropout rate for neural networks.
6. dispersion (Literal['gene', 'gene-batch'] (default: 'gene')) –
    One of the following:
    'gene' - dispersion parameter of NB is constant per gene across cells
    'gene-batch' - dispersion can differ between different batches
    'gene-label' - dispersion can differ between different labels
    'gene-cell' - dispersion can differ for every gene in every cell
7. gene_likelihood (Literal['nb', 'poisson'] (default: 'nb')) –
    One of:
    'nb' - Negative binomial distribution
    'zinb' - Zero-inflated negative binomial distribution
    'poisson' - Poisson distribution
8. **model_kwargs – Keyword args for VAE
## how to train the ResolVI model
RESOLVI.train(max_epochs=50, lr=0.003, lr_extra=0.01, extra_lr_parameters=('per_neighbor_diffusion_map', 'u_prior_means'), batch_size=512, weight_decay=0.0, eps=0.0001, n_steps_kl_warmup=None, n_epochs_kl_warmup=20, plan_kwargs=None, expose_params=(), **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 50)) – Number of passes through the dataset.
2. lr (float (default: 0.003)) – Learning rate for optimization.
3. lr_extra (float (default: 0.01)) – Learning rate for parameters (non-amortized and custom ones)
4. extra_lr_parameters (tuple (default: ('per_neighbor_diffusion_map', 'u_prior_means'))) – List of parameters to train with lr_extra learning rate.
5. batch_size (int (default: 512)) – Minibatch size to use during training.
6. weight_decay (float (default: 0.0)) – weight decay regularization term for optimization
7. eps (float (default: 0.0001)) – Optimizer eps
8. n_steps_kl_warmup (int | None (default: None)) – Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1. Only activated when n_epochs_kl_warmup is set to None. If None, defaults to floor(0.75 * adata.n_obs).
9. n_epochs_kl_warmup (int | None (default: 20)) – Number of epochs to scale weight on KL divergences from 0 to 1. Overrides n_steps_kl_warmup when both are not None.
10. plan_kwargs (dict | None (default: None)) – Keyword args for PyroTrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
11. expose_params (list (default: ())) – List of parameters to train if running model in Arches mode.
12. **kwargs – Other keyword args for Trainer.
## how to get results from trained model
RESOLVI.predict(adata=None, indices=None, soft=False, batch_size=500, num_samples=30)
### the detail description of parameters:
1. adata (AnnData | None (default: None)) – AnnData object that has been registered via setup_anndata().
2. indices (Sequence[int] | None (default: None)) – Subsample AnnData to these indices.
3. soft (bool (default: False)) – If True, returns per class probabilities
4. batch_size (int | None (default: 500)) – Minibatch size for data loading into model. Defaults to scvi.settings.batch_size.
5. num_samples (int | None (default: 30)) – Samples to draw from the posterior for cell-type prediction.
## get latent embedding
RESOLVI.get_latent_representation(adata=None, indices=None, give_mean=True, mc_samples=1, batch_size=None, return_dist=False)
### the detail description of parameters:
1. adata (AnnData | None (default: None)) – AnnData object with equivalent structure to initial AnnData. If None, defaults to the AnnData object used to initialize the model.
2. indices (Sequence[int] | None (default: None)) – Indices of cells in adata to use. If None, all cells are used.
3. give_mean (bool (default: True)) – Give mean of distribution or sample from it.
4. mc_samples (int (default: 1)) – For consistency with scVI, this parameter is ignored.
5. batch_size (int | None (default: None)) – Minibatch size for data loading into model. Defaults to scvi.settings.batch_size.
6. return_dist (bool (default: False)) – Return the distribution parameters of the latent variables rather than their sampled values. If True, ignores give_mean and mc_samples.
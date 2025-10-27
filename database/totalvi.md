# totalVI
## integration of CITE-seq and scRNA-seq with totalVI
## The import form of the totalVI package
import scvi
## setup data function of totalVI
scvi.model.TOTALVI.setup_anndata(adata, protein_expression_obsm_key, protein_names_uns_key=None, batch_key=None, layer=None, size_factor_key=None, categorical_covariate_keys=None, continuous_covariate_keys=None, **kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object. Rows represent cells, columns represent features.
2. protein_expression_obsm_key (str) – key in adata.obsm for protein expression data.
3. protein_names_uns_key (str | None (default: None)) – key in adata.uns for protein names. If None, will use the column names of adata.obsm[protein_expression_obsm_key] if it is a DataFrame, else will assign sequential names to proteins.
4. batch_key (str | None (default: None)) – key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
5. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
6. size_factor_key (str | None (default: None)) – key in adata.obs for size factor information. Instead of using library size as a size factor, the provided size factor column will be used as offset in the mean of the likelihood. Assumed to be on linear scale.
7. categorical_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to categorical data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
8. continuous_covariate_keys (list[str] | None (default: None)) – keys in adata.obs that correspond to continuous data. These covariates can be added in addition to the batch covariate and are also treated as nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus, these should not be used for biologically-relevant factors that you do _not_ want to correct for.
## setup model function of totalVI
scvi.model.TOTALVI(adata, n_latent=20, gene_dispersion='gene', protein_dispersion='protein', gene_likelihood='nb', latent_distribution='normal', empirical_protein_background_prior=None, override_missing_proteins=False, **model_kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object that has been registered via setup_anndata().
2. n_latent (int (default: 20)) – Dimensionality of the latent space.
3. gene_dispersion (Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] (default: 'gene')) –
    One of the following:
        'gene' - genes_dispersion parameter of NB is constant per gene across cells
        'gene-batch' - genes_dispersion can differ between different batches
        'gene-label' - genes_dispersion can differ between different labels
4. protein_dispersion (Literal['protein', 'protein-batch', 'protein-label'] (default: 'protein')) –
    One of the following:
        'protein' - protein_dispersion parameter is constant per protein across cells
        'protein-batch' - protein_dispersion can differ between different batches NOT TESTED
        'protein-label' - protein_dispersion can differ between different labels NOT TESTED
5. gene_likelihood (Literal['zinb', 'nb'] (default: 'nb')) –
    One of:
        'nb' - Negative binomial distribution
        'zinb' - Zero-inflated negative binomial distribution
6. latent_distribution (Literal['normal', 'ln'] (default: 'normal')) –
    One of:
        'normal' - Normal distribution
        'ln' - Logistic normal distribution (Normal(0, I) transformed by softmax)
7. empirical_protein_background_prior (bool | None (default: None)) – Set the initialization of protein background prior empirically. This option fits a GMM for each of 100 cells per batch and averages the distributions. Note that even with this option set to True, this only initializes a parameter that is learned during inference. If False, randomly initializes. The default (None), sets this to True if greater than 10 proteins are used.
8. override_missing_proteins (bool (default: False)) – If True, will not treat proteins with all 0 expression in a particular batch as missing.
9. **model_kwargs – Keyword args for TOTALVAE
## how to train the totalVI model
scvi.model.TOTALVI.train(max_epochs=None, lr=0.004, accelerator='auto', devices='auto', train_size=0.9, validation_size=None, shuffle_set_split=True, batch_size=256, early_stopping=True, check_val_every_n_epoch=None, reduce_lr_on_plateau=True, n_steps_kl_warmup=None, n_epochs_kl_warmup=None, adversarial_classifier=None, datasplitter_kwargs=None, plan_kwargs=None, external_indexing=None, **kwargs)
### the detail description of parameters:
1. max_epochs (int | None (default: None)) – Number of passes through the dataset.
2. lr (float (default: 0.004)) – Learning rate for optimization.
3. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
4. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
5. train_size (float (default: 0.9)) – Size of training set in the range [0.0, 1.0].
6. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
7. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
8. batch_size (int (default: 256)) – Minibatch size to use during training.
9. early_stopping (bool (default: True)) – Whether to perform early stopping with respect to the validation set.
10. check_val_every_n_epoch (int | None (default: None)) – Check val every n train epochs. By default, val is not checked, unless early_stopping is True or reduce_lr_on_plateau is True. If either of the latter conditions are met, val is checked every epoch.
11. reduce_lr_on_plateau (bool (default: True)) – Reduce learning rate on plateau of validation metric (default is ELBO).
n_steps_kl_warmup (int | None (default: None)) – Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1. Only activated when n_epochs_kl_warmup is set to None. If None, defaults to floor(0.75 * adata.n_obs).
12. n_epochs_kl_warmup (int | None (default: None)) – Number of epochs to scale weight on KL divergences from 0 to 1. Overrides n_steps_kl_warmup when both are not None.
13. adversarial_classifier (bool | None (default: None)) – Whether to use adversarial classifier in the latent space. This helps mixing when there are missing proteins in any of the batches. Defaults to True is missing proteins are detected.
14. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
15. plan_kwargs (dict | None (default: None)) – Keyword args for AdversarialTrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
16. external_indexing (list[array] (default: None)) – A list of data split indices in the order of training, validation, and test sets. Validation and test set are not required and can be left empty.
17. **kwargs – Other keyword args for Trainer.
## how to get results from trained model
model.get_latent_representation()
## how to get results from trained model
model.get_protein_foreground_probability(adata=None, indices=None, transform_batch=None, protein_list=None, n_samples=1, batch_size=None, return_mean=True, return_numpy=None)
### the detail description of parameters:
1. adata (AnnData | None (default: None)) – AnnData object with equivalent structure to initial AnnData. If None, defaults to the AnnData object used to initialize the model.
2. indices (Sequence[int] | None (default: None)) – Indices of cells in adata to use. If None, all cells are used.
3. transform_batch (Sequence[int | float | str] | None (default: None)) –
Batch to condition on. If transform_batch is:
    None, then real observed batch is used
    int, then batch transform_batch is used
    List[int], then average over batches in list
4. protein_list (Sequence[str] | None (default: None)) – Return protein expression for a subset of genes. This can save memory when working with large datasets and few genes are of interest.
5. n_samples (int (default: 1)) – Number of posterior samples to use for estimation.
6. batch_size (int | None (default: None)) – Minibatch size for data loading into model. Defaults to scvi.settings.batch_size.
7. return_mean (bool (default: True)) – Whether to return the mean of the samples.
8. return_numpy (bool | None (default: None)) – Return a ndarray instead of a DataFrame. DataFrame includes gene names as columns. If either n_samples=1 or return_mean=True, defaults to False. Otherwise, it defaults to True.
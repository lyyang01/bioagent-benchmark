# gimVI
## The import form of the gimVI package
import scvi
from scvi.external import GIMVI
## setup data function of gimVI
GIMVI.setup_anndata(adata, batch_key=None, labels_key=None, layer=None, **kwargs)
### the detail description of parameters:
1. batch_key (str | None (default: None)) – key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
2. labels_key (str | None (default: None)) – key in adata.obs for label information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_labels’]. If None, assigns the same label to all the data.
3. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
## setup model function of gimVI
GIMVI(adata_seq, adata_spatial, generative_distributions=None, model_library_size=None, n_latent=10, **model_kwargs)[source]
### the detail description of parameters:
1. adata_seq (AnnData) – AnnData object that has been registered via setup_anndata() and contains RNA-seq data.
2. adata_spatial (AnnData) – AnnData object that has been registered via setup_anndata() and contains spatial data.
n_hidden – Number of nodes per hidden layer.
3. generative_distributions (list[str] | None (default: None)) – List of generative distribution for adata_seq data and adata_spatial data. Defaults to [‘zinb’, ‘nb’].
4. model_library_size (list[bool] | None (default: None)) – List of bool of whether to model library size for adata_seq and adata_spatial. Defaults to [True, False].
5. n_latent (int (default: 10)) – Dimensionality of the latent space.
6. **model_kwargs – Keyword args for JVAE
## how to train the gimVI model
GIMVI.train(max_epochs=200, accelerator='auto', devices='auto', kappa=5, train_size=None, validation_size=None, shuffle_set_split=True, batch_size=128, datasplitter_kwargs=None, plan_kwargs=None, **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 200)) – Number of passes through the dataset. If None, defaults to np.min([round((20000 / n_cells) * 400), 400])
2. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
3. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
4. kappa (int (default: 5)) – Scaling parameter for the discriminator loss.
5. train_size (float | None (default: None)) – Size of training set in the range [0.0, 1.0].
6. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
7. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
8. batch_size (int (default: 128)) – Minibatch size to use during training.
9. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
10. plan_kwargs (dict | None (default: None)) – Keyword args for model-specific Pytorch Lightning task. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
11. **kwargs – Other keyword args for Trainer.
## how to get results from trained model
latent_seq, latent_spatial = model.get_latent_representation()
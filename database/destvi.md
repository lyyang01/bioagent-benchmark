# DestVI
## The import form of the DestVI related package
import scvi
from scvi.model import CondSCVI, DestVI
## You need to first setup data and train the model with CondSCVI, and then you setup data and train the model with DestVI
## Setup data with ConSCVI
CondSCVI.setup_anndata(adata, labels_key=None, layer=None, batch_key=None, **kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object. Rows represent cells, columns represent features.
2. labels_key (str | None (default: None)) – key in adata.obs for label information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_labels’]. If None, assigns the same label to all the data.
3. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
4. batch_key (str | None (default: None)) – key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
## Setup model with ConSCVI
scvi.model.CondSCVI(adata, n_hidden=128, n_latent=5, n_layers=2, weight_obs=False, dropout_rate=0.05, **module_kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object that has been registered via setup_anndata().
2. n_hidden (int (default: 128)) – Number of nodes per hidden layer.
3. n_latent (int (default: 5)) – Dimensionality of the latent space.
4. n_layers (int (default: 2)) – Number of hidden layers used for encoder and decoder NNs.
5. weight_obs (bool (default: False)) – Whether to reweight observations by their inverse proportion (useful for lowly abundant cell types)
6. dropout_rate (float (default: 0.05)) – Dropout rate for neural networks.
7. **module_kwargs – Keyword args for VAEC
## train the CondSCVI model
CondSCVI.train(max_epochs=300, lr=0.001, accelerator='auto', devices='auto', train_size=1, validation_size=None, shuffle_set_split=True, batch_size=128, datasplitter_kwargs=None, plan_kwargs=None, **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 300)) – Number of epochs to train for
2. lr (float (default: 0.001)) – Learning rate for optimization.
3. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
4. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
5. train_size (float (default: 1)) – Size of training set in the range [0.0, 1.0].
6. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
7. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
8. batch_size (int (default: 128)) – Minibatch size to use during training.
9. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
10. plan_kwargs (dict | None (default: None)) – Keyword args for TrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
11. **kwargs – Other keyword args for Trainer.
## Setup data with DestVI
DestVI.setup_anndata(adata, layer=None, **kwargs)
### the detail description of parameters:
1. adata (AnnData) – AnnData object. Rows represent cells, columns represent features.
2. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
## Setup model with DestVI from CondSCVI model
DestVI.from_rna_model(st_adata, sc_model, vamp_prior_p=15, l1_reg=0.0, **module_kwargs)
### the detail description of parameters:
1. st_adata (AnnData) – registered anndata object
2. sc_model (CondSCVI) – trained CondSCVI model
3. vamp_prior_p (int (default: 15)) – number of mixture parameter for VampPrior calculations
4. l1_reg (float (default: 0.0)) – Scalar parameter indicating the strength of L1 regularization on cell type proportions. A value of 50 leads to sparser results.
5. **model_kwargs – Keyword args for DestVI
## train the DestVI model
DestVI.train(max_epochs=2000, lr=0.003, accelerator='auto', devices='auto', train_size=1.0, validation_size=None, shuffle_set_split=True, batch_size=128, n_epochs_kl_warmup=200, datasplitter_kwargs=None, plan_kwargs=None, **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 2000)) – Number of epochs to train for
2. lr (float (default: 0.003)) – Learning rate for optimization.
3. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
4. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
5. train_size (float (default: 1.0)) – Size of training set in the range [0.0, 1.0].
6. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
7. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
8. batch_size (int (default: 128)) – Minibatch size to use during training.
9. n_epochs_kl_warmup (int (default: 200)) – number of epochs needed to reach unit kl weight in the elbo
10. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
11. plan_kwargs (dict | None (default: None)) – Keyword args for TrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
12. **kwargs – Other keyword args for Trainer.
## Returns the estimated cell type proportion for the spatial data.
DestVI.get_proportions(keep_noise=False, indices=None, batch_size=None)
### the detail description of parameters:
1. keep_noise (bool (default: False)) – whether to account for the noise term as a standalone cell type in the proportion estimate.
2. indices (Sequence[int] | None (default: None)) – Indices of cells in adata to use. Only used if amortization. If None, all cells are used.
3. batch_size (int | None (default: None)) – Minibatch size for data loading into model. Only used if amortization. Defaults to scvi.settings.batch_size.


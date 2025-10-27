# Stereoscope
## The import form of the Stereoscope package
import scvi
from scvi.external import RNAStereoscope, SpatialStereoscope
## You need to first setup data and train the model with RNAStereoscope, and then you setup data and train the model with SpatialStereoscope from RNAStereoscope model
## setup data with RNAStereoscope
RNAStereoscope.setup_anndata(adata, labels_key=None, layer=None, **kwargs)
### the detail description of parameters:
1. labels_key (str | None (default: None)) – key in adata.obs for label information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_labels’]. If None, assigns the same label to all the data.
2. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
## setup model with RNAStereoscope
scvi.external.RNAStereoscope(sc_adata, **model_kwargs)
### the detail description of parameters:
1. sc_adata (AnnData) – single-cell AnnData object that has been registered via setup_anndata().
2. **model_kwargs – Keyword args for RNADeconv
## train the RNAStereoscope model
RNAStereoscope.train(max_epochs=400, lr=0.01, accelerator='auto', devices='auto', train_size=1, validation_size=None, shuffle_set_split=True, batch_size=128, datasplitter_kwargs=None, plan_kwargs=None, **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 400)) – Number of epochs to train for
2. lr (float (default: 0.01)) – Learning rate for optimization.
3. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
4. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
5. train_size (float (default: 1)) – Size of training set in the range [0.0, 1.0].
6. validation_size (float | None (default: None)) – Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
7. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
8. batch_size (int (default: 128)) – Minibatch size to use during training.
9. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
10. plan_kwargs (dict | None (default: None)) – Keyword args for TrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
11. **kwargs – Other keyword args for Trainer.
## Setup data with SpatialStereoscope
SpatialStereoscope.setup_anndata(adata, layer=None, **kwargs)
### the detail description of parameters:
1. layer (str | None (default: None)) – if not None, uses this as the key in adata.layers for raw count data.
## Setup model with SpatialStereoscope from RNAStereoscope model
SpatialStereoscope.from_rna_model(st_adata, sc_model, prior_weight='n_obs', **model_kwargs)
### the detail description of parameters:
1. st_adata (AnnData) – registed anndata object
2. sc_model (RNAStereoscope) – trained RNADeconv model
3. prior_weight (Literal['n_obs', 'minibatch'] (default: 'n_obs')) – how to reweight the minibatches for stochastic optimization. “n_obs” is the valid procedure, “minibatch” is the procedure implemented in Stereoscope.
4. **model_kwargs – Keyword args for SpatialDeconv
## train the SpatialStereoscope model
SpatialStereoscope.train(max_epochs=400, lr=0.01, accelerator='auto', devices='auto', shuffle_set_split=True, batch_size=128, datasplitter_kwargs=None, plan_kwargs=None, **kwargs)
### the detail description of parameters:
1. max_epochs (int (default: 400)) – Number of epochs to train for
2. lr (float (default: 0.01)) – Learning rate for optimization.
3. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
4. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
5. shuffle_set_split (bool (default: True)) – Whether to shuffle indices before splitting. If False, the val, train, and test set are split in the sequential order of the data according to validation_size and train_size percentages.
6. batch_size (int (default: 128)) – Minibatch size to use during training.
7. datasplitter_kwargs (dict | None (default: None)) – Additional keyword arguments passed into DataSplitter.
8. plan_kwargs (dict | None (default: None)) – Keyword args for TrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
10. **kwargs – Other keyword args for Trainer.
## Returns the estimated cell type proportion for the spatial data.
SpatialStereoscope.get_proportions(keep_noise=False)
### the detail description of parameters:
1. keep_noise (default: False) – whether to account for the noise term as a standalone cell type in the proportion estimate.



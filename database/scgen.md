# scGEN
## The import form of the scGEN package
import scgen
## Sets up the AnnData object for this model.
SCGEN.setup_anndata(adata, batch_key=None, labels_key=None, **kwargs)
### the detail description of parameters:
1. batch_key : str | None (default: None)
key in adata.obs for batch information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_batch’]. If None, assigns the same batch to all the data.
2. labels_key : Optional[str] (default: None)
key in adata.obs for label information. Categories will automatically be converted into integer categories and saved to adata.obs[‘_scvi_labels’]. If None, assigns the same label to all the data.
## Build the scgen model
scgen.SCGEN(adata, n_hidden=800, n_latent=100, n_layers=2, dropout_rate=0.2, **model_kwargs)
### the detail description of parameters:
1. adata : AnnData
AnnData object that has been registered via setup_anndata().
2. n_hidden : int (default: 800)
Number of nodes per hidden layer.
3. n_latent : int (default: 100)
Dimensionality of the latent space.
4. n_layers : int (default: 2)
Number of hidden layers used for encoder and decoder NNs.
5. dropout_rate : float (default: 0.2)
Dropout rate for neural networks.
6. **model_kwargs
Keyword args for SCGENVAE
## Train the model.
SCGEN.train(max_epochs=None, use_gpu=None, train_size=0.9, validation_size=None, batch_size=128, early_stopping=False, plan_kwargs=None, **trainer_kwargs)
### the detail description of parameters:
1. max_epochs : int | None (default: None)
Number of passes through the dataset. If None, defaults to np.min([round((20000 / n_cells) * 400), 400])
2. use_gpu : str | int | bool | None (default: None)
Use default GPU if available (if None or True), or index of GPU to use (if int), or name of GPU (if str, e.g., ‘cuda:0’), or use CPU (if False).
3. train_size : float (default: 0.9)
Size of training set in the range [0.0, 1.0].
4. validation_size : float | None (default: None)
Size of the test set. If None, defaults to 1 - train_size. If train_size + validation_size < 1, the remaining cells belong to a test set.
5. batch_size : int (default: 128)
Minibatch size to use during training.
6. early_stopping : bool (default: False)
Perform early stopping. Additional arguments can be passed in **kwargs. See Trainer for further options.
7. plan_kwargs : dict | None (default: None)
Keyword args for TrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
8. **trainer_kwargs
Other keyword args for Trainer.
## Return the latent representation for each cell.
SCGEN.get_latent_representation(adata=None, indices=None, give_mean=True, mc_samples=5000, batch_size=None)
### the detail description of parameters:
1. adata : AnnData | None (default: None)
AnnData object with equivalent structure to initial AnnData. If None, defaults to the AnnData object used to initialize the model.
2. indices : Sequence[int] | None (default: None)
Indices of cells in adata to use. If None, all cells are used.
3. give_mean : bool (default: True)
Give mean of distribution or sample from it.
4. mc_samples : int (default: 5000)
For distributions with no closed-form mean (e.g., logistic normal), how many Monte Carlo samples to take for computing mean.
5. batch_size : int | None (default: None)
Minibatch size for data loading into model. Defaults to scvi.settings.batch_size.
## Predicts the cell type provided by the user in stimulated condition.
SCGEN.predict(ctrl_key=None, stim_key=None, adata_to_predict=None, celltype_to_predict=None, restrict_arithmetic_to='all')
### the detail description of parameters:
1. ctrl_key : basestring
key for control part of the data found in condition_key.
2. stim_key : basestring
key for stimulated part of the data found in condition_key.
3. adata_to_predict : ~anndata.AnnData
Adata for unperturbed cells you want to be predicted.
4. celltype_to_predict : basestring
The cell type you want to be predicted.
5. restrict_arithmetic_to : basestring or dict
Dictionary of celltypes you want to be observed for prediction.




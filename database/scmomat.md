# Scmomat
## The import form of the Scmomat package
import scmomat
Thank you for providing the usage examples. Based on these examples, I will now accurately describe the parameters and usage of the functions from the **scMoMaT** package.

## 1. `scmomat.preprocess()`
```python
scmomat.preprocess(counts_atac1, modality = "ATAC")
```
### Description:
Preprocesses the input data for scMoMaT analysis, tailored to the specified modality.
### Parameters:
1. **`counts_atac1`**: matrix or data frame  
   - Input data matrix or data frame, typically containing ATAC-seq data.
2. **`modality`**: str (default: "ATAC")  
   - Modality of the input data. Options include "ATAC", "RNA", etc.
## 2. `scmomat.scmomat_model()`
```python
scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
```
### Description:
Constructs the scMoMaT model for analyzing single-cell multi-omics data.
### Parameters:
1. **`counts`**: matrix or data frame  
   - Preprocessed data matrix or data frame.
2. **`K`**: int  
   - Number of latent dimensions for the model.
3. **`batch_size`**: int  
   - Batch size for training.
4. **`interval`**: int  
   - Interval for logging training progress.
5. **`lr`**: float  
   - Learning rate for the optimizer.
6. **`lamb`**: float  
   - Regularization parameter.
7. **`seed`**: int  
   - Random seed for reproducibility.
8. **`device`**: str  
   - Device to use for training (e.g., "cpu" or "cuda").
## 3. `model.train_func()`
```python
model.train_func(T = T)
```
### Description:
Trains the scMoMaT model using the provided data.
### Parameters:
1. **`T`**: int  
   - Number of training epochs.
## 4. `scmomat.plot_latent()`
```python
scmomat.plot_latent(x_umap, annos = np.concatenate(labels_batches), mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
```
### Description:
Plots the latent space representation of the data.

### Parameters:
1. **`x_umap`**: matrix or data frame  
   - Latent space representation of the data.
2. **`annos`**: array-like  
   - Annotations for the data points.
3. **`mode`**: str (default: "joint")  
   - Mode for plotting. Options include "joint" for joint visualization.
4. **`save`**: str or None (default: None)  
   - File path to save the plot. If None, the plot is not saved.
5. **`figsize`**: tuple (default: (15,10))  
   - Figure size for the plot.
6. **`axis_label`**: str (default: "UMAP")  
   - Label for the axes.
7. **`markerscale`**: int (default: 6)  
   - Scale for markers in the legend.
8. **`s`**: int (default: 5)  
   - Size of the markers.
9. **`label_inplace`**: bool (default: True)  
   - Whether to label the data points in place.
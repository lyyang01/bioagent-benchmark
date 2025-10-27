# MIRA 


## Overview

MIRA is a library for analyzing single-cell RNA-seq or ATAC-seq data using topic modeling approaches. It learns regulatory "topics" that capture patterns of covariance between genes or cis-regulatory elements.

## Topics Module
import mira
### make_model

```python
mira.topics.make_model(
    n_samples, 
    n_features, 
    *, 
    feature_type, 
    highly_variable_key=None, 
    exogenous_key=None, 
    endogenous_key=None, 
    counts_layer=None, 
    categorical_covariates=None, 
    continuous_covariates=None, 
    covariates_keys=None, 
    extra_features_keys=None, 
    **model_parameters
)
```

Instantiates a topic model, which learns regulatory "topics" from single-cell RNA-seq or ATAC-seq data. Topics capture patterns of covariance between gene or cis-regulatory elements. Each cell is represented by a composition over topics, and each topic corresponds with activations of co-regulated elements.

When working with batched data, the parameters of the topic model are optimized using the novel CODAL (COvariate Disentangling Augmented Loss) objective, which shows State of the Art performance for detection of batch confounded cell types.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | `int` | Number of samples in the dataset, used to choose hyperparameters for the model |
| `n_features` | `int` | Number of features in the dataset, used to choose hyperparameters for the model |
| `feature_type` | `{'expression', 'accessibility'}` | Modality of the data being modeled |
| `highly_variable_key` | `str`, default=`None` | Column in AnnData that marks features to be modeled. These features should include all elements used for enrichment analysis of topics. For expression data, this should be highly variable genes relevant to your system (the top ~4000 appears to work well). For accessibility data, all called peaks may be used. |
| `exogenous_key` | `str`, default=`None` | Same as highly_variable_key, included for backwards compatibility |
| `endogenous_key` | `str`, default=`None` | Column in AnnData that marks features to be used for encoder neural network. These features should prioritize elements that distinguish between populations, like highly-variable genes. If "None", then the model will use the features supplied to "exogenous_key" |
| `counts_layer` | `str`, default=`None` | Layer in AnnData that contains raw counts for modeling |
| `categorical_covariates` | `str`, `list[str]`, `np.ndarray[str]`, or `None`, default=`None` | Categorical covariates in the dataset. For example, batch of origin, donor, assay chemistry, sequencing machine, etc. |
| `continuous_covariates` | `str`, `list[str]`, `np.ndarray[str]`, or `None`, default=`None` | Continuous covariates in the dataset. For example, FRIP score (ATAC-seq), percent reads mitochondria (RNA-seq), or other QC metrics |
| `extra_features_keys` | `str`, `list[str]`, `np.ndarray[str]`, or `None`, default=`None` | Columns in anndata.obs which contain extra features for the encoder neural network |

#### Returns

| Return Type | Description |
|-------------|-------------|
| `mira.topics.TopicModel` | A CODAL (if there are technical covariates in the dataset) or MIRA topic model. Hyperparameters of the topic model are chosen based on the supplied dataset properties |

#### Other Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cost_beta` | `float>0`, default=`1.` | Multiplier of the regularization loss terms (KL divergence and mutual information regularization) versus the reconstruction loss term. Smaller datasets (<10K cells) sometimes require larger cost_beta (1.25-2.), while larger datasets (>10K cells) always work well with cost_beta=1. This parameter is automatically set to a reasonable value based on the size of the dataset provided to this function |
| `num_topics` | `int`, default=`16` | Number of topics to learn from data |
| `hidden` | `int`, default=`128` | Number of nodes to use in hidden layers of encoder network |
| `num_layers` | `int`, default=`3` | Number of layers to use in encoder network, including output layer |
| `num_epochs` | `int`, default=`40` | Number of epochs to train topic model. The One-cycle learning rate policy requires a pre-defined training length, and 40 epochs is usually an overestimate of the optimal number of epochs to train for |
| `decoder_dropout` | `float (0., 1.)`, default=`0.2` | Dropout rate for the decoder network. Prevents node collapse |
| `encoder_dropout` | `float (0., 1.)`, default=`0.2` | Dropout rate for the encoder network. Prevents overfitting |
| `use_cuda` | `boolean`, default=`True` | Try using CUDA GPU speedup while training |
| `seed` | `int`, default=`None` | Random seed for weight initialization. Enables reproducible initialization of model |
| `min_learning_rate` | `float`, default=`1e-6` | Start learning rate for One-cycle learning rate policy |
| `max_learning_rate` | `float`, default=`1e-1` | Peak learning rate for One-cycle policy |
| `batch_size` | `int`, default=`64` | Minibatch size for stochastic gradient descent while training. Larger batch sizes train faster, but may produce less optimal models |
| `initial_pseudocounts` | `int`, default=`50` | Initial pseudocounts allocated to approximated hierarchical dirichlet prior. More pseudocounts produces smoother topics, less pseudocounts produces sparser topics |
| `nb_parameterize_logspace` | `boolean`, default=`True` | Parameterize negative-binomial distribution using log-space probability estimates of gene expression. Is more numerically stable |
| `embedding_size` | `int > 0 or None`, default=`None` | Number of nodes in first encoder neural network layer. Default of None gives an embedding size of hidden |
| `kl_strategy` | `{'monotonic', 'cyclic'}`, default=`'cyclic'` | Whether to anneal KL term using monotonic or cyclic strategies. Cyclic may produce slightly better models |

#### CODAL Models Only Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dependence_lr` | `float>0`, default=`1e-4` | Learning rate for tuning the mutual information estimator |
| `dependence_hidden` | `int>0`, default=`64` | Hidden size of mutual information estimator |
| `weight_decay` | `float>0`, default=`0.001` | Weight decay of topic model weight optimizer |
| `min_momentum` | `float>0`, default=`0.85` | Min momentum for 1-cycle learning rate policy |
| `max_momentum` | `float>0`, default=`0.95` | Max momentum for 1-cycle learning rate policy |
| `covariates_hidden` | `int>0`, default=`32` | Number of nodes for single layer of technical effect network |
| `covariates_dropout` | `float>0`, default=`0.05` | Dropout applied to the technical effect network |
| `mask_dropout` | `float>0`, default=`0.05` | Bernoulli corruption rate of technical effect predictions during training |
| `marginal_estimation_size` | `int>0`, default=`256` | Number of pairings used to estimate mutual information at each step |
| `dependence_beta` | `float>0`, default=`1.` | The weight of the mutual information cost at each step is cost_beta*dependence_beta. Changing this value to more than 1 weights mutual information regularization more highly than KL-divergence regularization of the loss |

#### Accessibility Models Only Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding_dropout` | `float>0`, default=`0.05` | Bernoulli corruption of bag of peaks input to DAN encoder |
| `atac_encoder` | `str in {"fast", "skipDAN", "DAN"}`, default=`"skipDAN"` | Which type of ATAC encoder to use. The best results are given by "skipDAN", which is the default. However, this model is pretty much impossible to train on CPU. If instantiated without GPU, will throw an error and suggest the "fast" encoder. The "fast" encoder skips the large embedding layer of the DAN models and calculates a first-pass LSI projection of the data |

#### Example

```python
model = mira.topics.make_model(
    *rna_data.shape,
    feature_type='expression',
    highly_variable_key='highly_variable', 
    counts_layer='rawcounts',
    categorical_covariates=['batch', 'donor'],
    continuous_covariates=['FRIP']
)
```

### gradient_tune

```python
mira.topics.gradient_tune(model, data, max_attempts=3, max_topics=None)
```

Tune number of topics using a gradient-based estimator based on the Dirichlet Process model. This tuner is very fast, though less comprehensive than the BayesianTuner. We recommend using this tuner for large datasets (>40K cells).

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `mira.topics.TopicModel` | Topic model to tune. The provided model should have columns specified to retrieve endogenous and exogenous features, and should have the learning rate configured by get_learning_rate_bounds |
| `data` | `anndata.AnnData` | Anndata of expression or accessibility data |
| `max_attempts` | `int`, default=`3` | Maximum number of attempts for tuning |
| `max_topics` | `int > 0 or None`, default=`None` | If None, MIRA automatically chooses an upper limit on the number of topics to model based on a generous heuristic calculated from the number of cells in the provided dataset. If a value is provided, that upper limit is used instead |

#### Returns

| Return Type | Description |
|-------------|-------------|
| `int` | Estimated number of topics in dataset |
| `np.ndarray[float] of shape (n_topics,)` | For each topic attempted to learn from the data, its maximum contribution to any cell |

### BayesianTuner

```python
class mira.topics.BayesianTuner(
    *, 
    model, 
    save_name, 
    min_topics, 
    max_topics, 
    storage='sqlite:///mira-tuning.db', 
    n_jobs=1, 
    max_trials=128, 
    min_trials=48, 
    stop_condition=12, 
    seed=2556, 
    tensorboard_logdir='runs', 
    model_dir='models', 
    pruner=None, 
    sampler=None, 
    log_steps=False, 
    log_every=10, 
    train_size=0.8
)
```

A BayesianTuner object chooses the number of topics and the appropriate regularization to produce a model that best fits the user's dataset.

The process consists of iteratively training a model using a set of hyperparameters, evaluating the resulting model, and choosing the next set of hyperparameters based on which set is most likely to yield an improvement over previous models trained.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `mira.topics.TopicModel` | Topic model to tune. The provided model should have columns specified to retrieve endogenous and exogenous features, and should have the learning rate configured by get_learning_rate_bounds |
| `save_name` | `str (required)` | Table under which to save tuning results in storage table. A good pattern to follow is: dataset/modality/model_id/tuning_run |
| `min_topics` | `int (required)` | Minimum number of topics to try |
| `max_topics` | `int (required)` | Maximum number of topics to try |
| `storage` | `str or mira.topics.Redis()`, default=`'sqlite:///mira-tuning.db'` | The default value saves the results from tuning in an SQLite table with the file location ./mira-tuning.db. SQLite tables require no outside libraries, but can only handle read-write for up to 5 concurrent processes. Tuning can be significantly sped up by running even more concurrent processes, which requires a REDIS database backend with faster read-write speeds. To use the REDIS backend, start a REDIS server in the background, and pass a mira.topics.Redis() object to this parameter. Adjust the url as needed |
| `n_jobs` | `int>0`, default=`1` | Number of concurrent trials to run at a time. The default SQLite backend can handle up to 5, but the REDIS backend can handle many more (>20!). Each trial's memory footprint is essentially that of the model parameters, the optimizer, and one batch of training (because the dataset is saved to disk and streamed batch-by-batch during model training). Thus, training a model with 200K cells requires the same memory as on 1000 cells. We suggest taking advantage of the low memory overhead to train concurrently across as many cores as possible |
| `tensorboard_logdir` | `str`, default=`'runs'` | Directory in which to save tensorboard log files |
| `min_trials` | `int>0`, default=`48` | Minimum number of trials to run |
| `max_trials` | `int>0`, default=`128` | If finding better models, continues to train until reaching this number of trials |
| `stop_condition` | `int>0`, default=`12` | Continue tuning until a better model has not been produced for this many iterations |
| `seed` | `int`, default=`2556` | Random seed for reproducibility |
| `model_dir` | `path`, default=`'./models/'` | Where to save the best models trained during tuning |
| `pruner` | `None or optuna.pruners.Pruner`, default=`None` | If None, uses the default SuccessiveHalving bandit pruner |
| `sampler` | `None or optuna.pruner.BaseSampler`, default=`None` | If None, uses MIRA's default choice of a Gaussian Process sampler with pruning |
| `log_steps` | `boolean`, default=`False` | Whether to save loss at every step of training. Useful for debugging, but slows down tuning |
| `log_every` | `int`, default=`10` | How often to log during training |
| `train_size` | `float`, default=`0.8` | Fraction of data to use for training (vs. validation) |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `study` | `optuna.study.Study` | Optuna study object summarizing tuning results |
| `trial_attrs` | `list of dicts` | Data for each trial |

#### Methods

##### load

```python
@classmethod
load(*, model, save_name, storage='sqlite:///mira-tuning.db')
```

Load a tuning run from the given storage object.

##### purge

```python
purge()
```

If tuning is stopped with some trials in progress, those trials will be saved as "zombie" trials, doomed never to be completed. Upon restart of tuning, those zombie trials can interfere with selection of hyperparameters.

This function changes the state of all RUNNING trials to FAILED.

##### fit

```python
fit(train, test=None)
```

Run Bayesian optimization scheme for topic model hyperparameters. This function launches multiple concurrent training processes to evaluate hyperparameter combinations. All processes are launched on the same node. Evaluate the memory usage of a single MIRA topic model to determine number of workers.

###### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `train` | `anndata.AnnData` | Anndata of expression or accessibility data. If test is not provided, this dataset will be partitioned into train and test sets according to the ratio given by the train_size parameter |
| `test` | `anndata.AnnData`, default=`None` | Anndata of expression or accessibility data. Evaluation set of cells |

###### Returns

| Return Type | Description |
|-------------|-------------|
| `mira.topics.TopicModel` | Topic model trained with best set of hyperparameters found during tuning |

##### fetch_weights

```python
fetch_weights(trial_num)
```

Fetch topic model weights trained in the given trial from disk. Can only fetch weights from trials which were not pruned.

###### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `trial_num` | `int` | Trial number for which to fetch weights |

###### Returns

| Return Type | Description |
|-------------|-------------|
| `mira.topics.TopicModel` | The topic model from the specified trial |

###### Raises

| Exception | Description |
|-----------|-------------|
| `ValueError` | If trial does not exist |
| `KeyError` | If trial did not finish |

##### fetch_best_weights

```python
fetch_best_weights()
```

Fetch weights best topic model trained during tuning. This is the "official" topic model for a given dataset.

###### Returns

| Return Type | Description |
|-------------|-------------|
| `mira.topics.TopicModel` | The best topic model found during tuning |

###### Raises

| Exception | Description |
|-----------|-------------|
| `ValueError` | If no trials have been completed |

##### plot_intermediate_values

```python
plot_intermediate_values(
    palette='Spectral_r', 
    hue='value', 
    ax=None, 
    figsize=(10, 7), 
    log_hue=False, 
    na_color='lightgrey', 
    add_legend=True, 
    vmax=None, 
    vmin=None, 
    **plot_kwargs
)
```

Plots the evaluation loss achieved at each epoch of training for all of the trials.

###### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `palette` | `str`, default=`'Spectral_r'` | Which color to plot for each trial |
| `ax` | `matplotlib.pyplot.axes or None`, default=`None` | Provide axes object to function for more control. If no axes are provided, they are created internally |
| `figsize` | `tuple[int, int]`, default=`(10,7)` | Size of plot |
| `log_hue` | `boolean`, default=`False` | Take the log of the hue value to plot |
| `hue` | `{'value', 'number', 'num_topics', 'decoder_dropout', 'rate', 'distortion', … }`, default=`"value"` | Which attribute of each trial to plot. For a full list of attributes, use: tuner.trial_attrs. The default "value" is the objective score |
| `vmin`, `vmax` | `float`, default=`None` | Minimum and maximum bounds on continuous color palettes |
| `na_color` | `str`, default=`'lightgrey'` | Color for NA values |
| `add_legend` | `boolean`, default=`True` | Whether to add a legend to the plot |

###### Returns

| Return Type | Description |
|-------------|-------------|
| `matplotlib.pyplot.axes` | The plot axes |

##### plot_pareto_front

```python
plot_pareto_front(
    x='num_topics', 
    y='elbo', 
    hue='number', 
    ax=None, 
    figsize=(7, 7), 
    palette='Blues', 
    na_color='lightgrey', 
    size=100, 
    alpha=0.8, 
    add_legend=True, 
    label_pareto_front=False, 
    include_pruned_trials=True
)
```

Relational plot of tuning trials data. Often, it is most interesting to compare the objective value ("elbo") versus the number of topics. This serves as a sanity check that the objective is convex with respect to topics and that the tuner converged on the appropriate number of topics for the dataset.

###### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `str`, default=`"num_topics"` | Trial attribute to plot on x-axis. Use tuner.trial_attrs to see list of possible attributes to plot |
| `y` | `str`, default=`"elbo"` | Trial attribute to plot on y-axis. "elbo" and "value" plot the objective score. "distortion" plots the reconstruction loss. "rate" plots the KL-divergence loss |
| `hue` | `{'value', 'number', 'num_topics', 'decoder_dropout', 'rate', 'distortion', … }`, default=`"number"` | Which attribute of each trial to plot. For a full list of attributes, use: tuner.trial_attrs |
| `ax` | `matplotlib.pyplot.axes or None`, default=`None` | Provide axes object to function for more control. If no axes are provided, they are created internally |
| `figsize` | `tuple[int, int]`, default=`(7,7)` | Size of plot |
| `palette` | `str`, default=`'Blues'` | Color palette to use |
| `na_color` | `str`, default=`'lightgrey'` | Color for NA values |
| `size` | `int`, default=`100` | Size of plot points |
| `alpha` | `float`, default=`0.8` | Transparency of plot points |
| `add_legend` | `boolean`, default=`True` | Whether to add a legend to the plot |
| `label_pareto_front` | `boolean`, default=`False` | Only label trials on the pareto front of distortion and rate, e.g. the best trials |
| `include_pruned_trials` | `boolean`, default=`True` | Whether to include pruned trials in the plot |

###### Returns

| Return Type | Description |
|-------------|-------------|
| `matplotlib.pyplot.axes` | The plot axes |

#### Example

```python
tuner = mira.topics.BayesianTuner(
    model=model,
    min_topics=5,
    max_topics=55,
    n_jobs=1,
    save_name='tuning/rna/0',
)
tuner.fit(data)
model = tuner.fetch_best_weights()
```
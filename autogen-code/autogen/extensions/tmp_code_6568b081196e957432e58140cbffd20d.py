# Import basic data manipulation and visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import scanpy for single-cell analysis
import scanpy as sc
import anndata as ad

# Import scvi-tools for PeakVI model
import scvi

# Set random seed for reproducibility
np.random.seed(42)

# Set scanpy settings
sc.settings.verbosity = 1  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=100, facecolor='white')

# Import additional libraries that might be useful for scATAC-seq analysis
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# For parallel processing (optional)
import multiprocessing
n_cpus = multiprocessing.cpu_count()
print(f"Number of available CPUs: {n_cpus}")

# Check scvi-tools version
print(f"scvi-tools version: {scvi.__version__}")
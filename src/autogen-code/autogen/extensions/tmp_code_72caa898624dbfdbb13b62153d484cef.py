# Define the input data path
input_path = '/mnt/data00/share_data/agent_benchmark/peakvi/atac_pbmc_5k.untar/filtered_peak_bc_matrix'

# Load the 10X scATAC-seq data using scanpy's read_10x_mtx function
# For ATAC-seq data, the matrix contains peak x cell information
print(f"Loading 10X scATAC-seq data from: {input_path}")
adata = sc.read_10x_mtx(
    path=input_path,
    var_names='gene_symbols',  # Use gene_symbols as variable names if available
    cache=True                 # Cache the data for faster reloading
)

# Print basic information about the loaded data
print(f"Loaded AnnData object with {adata.shape[0]} cells and {adata.shape[1]} peaks")

# Check if the data is sparse (which is typical for scATAC-seq data)
print(f"Data matrix type: {type(adata.X)}")

# Check the sparsity of the data (percentage of non-zero entries)
sparsity = 100 * (1 - adata.X.nnz / (adata.shape[0] * adata.shape[1])) if sp.issparse(adata.X) else 0
print(f"Data sparsity: {sparsity:.2f}% (percentage of zeros)")

# Basic quality control metrics
print("\nComputing basic QC metrics...")
# Calculate number of peaks per cell
adata.obs['n_peaks'] = np.array(adata.X.sum(axis=1)).flatten()
# Calculate number of cells per peak
adata.var['n_cells'] = np.array(adata.X.sum(axis=0)).flatten()

# Show a summary of the data
print("\nSummary of peaks per cell:")
print(adata.obs['n_peaks'].describe())

# Ensure the matrix is in the correct format for PeakVI (CSR sparse matrix)
if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
    print("Converting matrix to CSR format for better performance with PeakVI...")
    adata.X = adata.X.tocsr()

# Save a copy of the raw data
adata.raw = adata.copy()

print("\nData loading complete and ready for PeakVI analysis")
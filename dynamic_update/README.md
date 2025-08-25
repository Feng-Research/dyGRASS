# dyGRASS Dynamic Implementation

Enhanced unified implementation of dyGRASS — Dynamic Graph Spectral Sparsification via Localized Random Walks. This improved version provides a single executable that handles both incremental and decremental graph updates with optimized memory management and enhanced user experience.

## Key Features

- **Unified Pipeline**: Single executable handles both incremental and decremental updates seamlessly
- **Memory Optimized**: Efficient GPU memory management with batch processing and overflow handling
- **Performance Enhanced**: Streamlined data structures and optimized CUDA kernels
- **Interactive Processing**: Real-time condition number analysis and professional output formatting
- **Comprehensive Timing**: Individual kernel timing and total runtime reporting
- **Production Ready**: Error handling, result validation, and automated dataset management

## Quick Start

### Prerequisites
- **Linux** (tested on Ubuntu 22.04)
- **CUDA 11.x or newer** with `nvcc` in `$PATH`
- **Python 3** for dataset download
- **Julia** for condition number analysis

### 1. Setup Environment
```bash
# Create virtual environment (optional but recommended)
cd .. && uv venv && cd dynamic_update/

# Install dependencies for dataset download
/home/yihang/release/dyGRASS/.venv/bin/python -m pip install gdown
```

### 2. Download Datasets
```bash
# Download and extract all datasets (238MB)
/home/yihang/release/dyGRASS/.venv/bin/python datasetdownload.py

# Alternative: Manual download
# Download dataset.tar.xz from Google Drive and extract to ./dataset/
```

### 3. Compile
```bash
# Compile the unified dynamic implementation
nvcc -ccbin g++-13 -g -G main.cu functions.cpp -o debug
```

### 4. Run Experiments
```bash
# Basic usage
./debug <graph_name> <distortion_threshold>

# Examples
./debug G2 100              # Small test graph
./debug fe_4elt 100         # Finite element mesh
./debug AS365 100           # Network topology
```

## Available Datasets

The implementation includes 14 ready-to-use datasets:

### Network/Topology Graphs
- **333SP** - Small-world network
- **AS365** - Autonomous system topology  
- **G2**, **G3** - Graph benchmark instances

### Finite Element Meshes
- **M6** - Mesh benchmark
- **NACA** - Airfoil mesh
- **fe_4elt** - 4-element finite element
- **fe_ocean** - Ocean mesh
- **fe_sphere** - Spherical mesh

### Deletion Test Sets
- **del18** through **del22** - Edge deletion benchmarks

## File Structure

Each dataset contains:
```
dataset/<graph_name>/
├── new_adj_dense.mtx      # Dense graph representation
├── new_adj_sparse.mtx     # Initial sparse approximation
└── stream_edges/          # Batch update files
    ├── stream_0000_incremental.mtx  # Incremental batches (0-9)
    ├── ...
    ├── stream_0009_incremental.mtx
    ├── stream_0010_decremental.mtx  # Decremental batches (10-19)
    ├── ...
    └── stream_0019_decremental.mtx
```

## Program Flow

1. **Initialization**: Load dense and sparse graphs, initialize GPU memory
2. **Batch Processing**: Process edge updates sequentially
   - Load next batch from stream files
   - Execute GPU random walk kernels (incremental/decremental)
   - Update sparsifier based on results
   - Optional: Check graph properties and condition numbers
3. **Results**: Generate timestamped output with final graphs

## Interactive Features

- **Property Checking**: Option to analyze graph properties after each batch
- **Condition Number Analysis**: Real-time spectral analysis via Julia integration
- **Timing Reports**: Individual kernel execution times and total runtime
- **Professional Output**: Formatted batch headers and progress indicators

## Output

Results are saved to `./output/<graph_name>/<timestamp>/`:
- `adj_dense.mtx` - Final dense graph
- `adj_sparse.mtx` - Final sparse approximation

## Command Line Options

```bash
./debug <graph_name> <distortion_threshold> [inc_steps] [dec_steps] [inc_walkers] [dec_walkers]
```

**Parameters:**
- `graph_name`: Dataset name (e.g., G2, fe_4elt)
- `distortion_threshold`: Spectral similarity bound (typically 100)
- `inc_steps`: Random walk steps for incremental (default: 100)
- `dec_steps`: Random walk steps for decremental (default: 100) 
- `inc_walkers`: Walkers for incremental (default: 512)
- `dec_walkers`: Walkers for decremental (default: 512)

## Technical Details

### Algorithm
- **Incremental**: Sample random walks to find paths between edge endpoints; add edges NOT found in walks to sparse graph
- **Decremental**: Find replacement paths when edges are removed to maintain connectivity
- **GPU Execution**: 512 walkers per edge with block-parallel execution (1 block per edge)

### Memory Management
- Dual CSR representation (dense + sparse graphs)
- Dynamic edge insertion/deletion with O(1) hash mapping
- Batch processing with overflow handling for large datasets
- Efficient GPU-CPU data transfer mechanisms

### Integration
- **Julia Interface**: Automatic condition number computation
- **File I/O**: Memory-mapped file access for large graphs
- **Error Handling**: Comprehensive CUDA error checking and recovery

## Performance

The unified implementation provides significant improvements over separate incremental/decremental programs:
- **Memory efficiency**: Shared data structures reduce GPU memory usage
- **Processing speed**: Streamlined batch workflow eliminates redundant operations
- **User productivity**: Single compilation and execution step

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Use smaller batch sizes or reduce walker count
- **File not found**: Verify dataset directory structure and file permissions
- **Compilation errors**: Ensure `nvcc` is in PATH and CUDA toolkit is properly installed
- **Julia errors**: Install required packages: Laplacians.jl, SparseArrays.jl, Arpack.jl

### Debug Mode
The executable is compiled with debug information (`-g -G`) for development and troubleshooting.

## Citation

```bibtex
@inproceedings{yuan2025dygrass,
  author    = {Yihang Yuan, Ali Aghdaei and Zhuo Feng},
  title     = {{dyGRASS}: Dynamic Spectral Graph Sparsification via Localized Random Walks on GPUs},
  booktitle = {Proceedings of the IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  year      = {2025}
}
```

## License

dyGRASS is released under the **MIT License**.
# dyGRASS

Code for dyGRASS — Dynamic Graph Spectral Sparsification via Localized Random Walks is the open-source reference implementation of the framework introduced in our ICCAD 2025 paper. dyGRASS updates spectral sparsifiers in O(1) time per edge insertion or deletion by launching GPU-accelerated non-backtracking random walks that pinpoint spectrally critical edges; this yields ~10 × speed-ups over the previous inGRASS algorithm while preserving tight spectral similarity across fully dynamic graphs.

## Setup

### System
* **Linux 22.04** (tested)  
* **CUDA 11.x or newer** with `nvcc` in `$PATH`

### 1. Grab the dataset
```bash
pip install  gdown         
python3 datasetdownload.py    # downloads + unpacks into ./dataset/     
```

### 2.Build the CUDA random-walk updaters

Prerequisite — CUDA toolkit 11.0 or newer (make sure `nvcc` is on your `$PATH`).

```bash
# Compile the incremental updater
cd random_walk_incremental
nvcc main.cu functions.cpp -o incremental_update
cd ..

# Compile the decremental updater
cd ../random_walk_decremental
nvcc main.cu functions.cpp -o decremental_update
cd ..
```
### 3. Julia environment for result reproduction
Open Julia in the project root and add the required packages:
```bash
using Pkg
Pkg.add([
    "Laplacians",
    "SparseArrays",
    "LinearAlgebra",
    "Arpack",
    "JLD2",
    "IterativeSolvers",
    "Statistics",
])
```

## Input File Requirements

### File Format
All graph files must be in **Matrix Market (.mtx) format**:
- **1-based indexing** (automatically converted to 0-based internally)
- Standard MTX header (optional, will be skipped)
- Format: `source destination [weight]`

### Required Files per Dataset
```
dataset/<graph_name>/
├── adj_sparse.mtx     # Initial sparse graph (1-based, weighted)
├── ext.mtx           # Extension edges for incremental (1-based, weighted)  
├── dense.mtx         # Dense graph for decremental (1-based, weighted)
├── del.mtx          # Deletion candidates for decremental (1-based, unweighted)
└── updated_dense.mtx # Updated dense graph (1-based, weighted)
```

### File Specifications
- **Weighted files** (`adj_sparse.mtx`, `ext.mtx`, `dense.mtx`, `updated_dense.mtx`): Include edge weights
- **Unweighted files** (`del.mtx`): No weights, format: `source destination`
- **Headers**: MTX headers are automatically detected and skipped
- **Index Format**: 1-based indexing in files (converted to 0-based internally)

## Usage

```bash
./random_walk_incremental/incremental_update <graph_name> <distortion_threshold> <#walkers>
# example
./random_walk_incremental/incremental_update G2 10 512
```
- graph_name  base name of the graph files inside dataset/
- distortion_threshold spectral-similarity bound (e.g. 100)
- walkers  number of non-backtracking walkers (≤ 512, multiple of 32)


```bash
./random_walk_decremental/decremental_update <graph_name>
# example
./random_walk_decremental/decremental_update 333SP
```

### Common Issues
- Ensure all vertex IDs are consecutive starting from 1
- Check file permissions for dataset directory
- Verify CUDA GPU memory is sufficient for large graphs
- Make sure `nvcc` is in your PATH for compilation

## Citing


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
See \`LICENSE\` for the full text.
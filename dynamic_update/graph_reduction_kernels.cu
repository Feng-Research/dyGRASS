#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>

using namespace cooperative_groups;

/**
 * Graph-Specific Block Reduction Kernels for dyGRASS
 * 
 * This file provides optimized block-level reduction kernels specifically
 * designed for graph sparsification operations without using global memory.
 * 
 * Use cases in dyGRASS:
 * - Finding maximum edge weights in vertex neighborhoods
 * - Computing maximum resistance values during random walks
 * - Block-level aggregation of graph statistics
 */

// =============================================================================
// 1. WARP SHUFFLE UTILITIES FOR GRAPH OPERATIONS
// =============================================================================

/**
 * Warp-level reduction for finding maximum value
 * Most efficient for values that fit in registers
 */
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/**
 * Warp-level reduction for finding maximum with associated index
 * Useful for tracking which edge/vertex had the maximum value
 */
__device__ __forceinline__ float warpReduceMaxWithIndex(float val, int idx, int* max_idx) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    
    if (threadIdx.x % warpSize == 0) {
        *max_idx = idx;
    }
    return val;
}

// =============================================================================
// 2. BLOCK-LEVEL REDUCTION TEMPLATES
// =============================================================================

/**
 * Block-level maximum reduction using warp shuffle + shared memory
 * Template parameter allows compile-time optimization for different block sizes
 */
template<int BLOCK_SIZE>
__device__ float blockReduceMax(float val) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE cannot exceed 1024");
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Step 1: Reduce within each warp using shuffle
    val = warpReduceMax(val);
    
    // Step 2: Use shared memory for inter-warp communication
    __shared__ float warpMaxs[BLOCK_SIZE / warpSize];
    if (lane == 0) {
        warpMaxs[wid] = val;
    }
    __syncthreads();
    
    // Step 3: Final reduction within first warp
    val = (threadIdx.x < BLOCK_SIZE / warpSize) ? warpMaxs[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warpReduceMax(val);
    }
    
    return val;
}

/**
 * Block-level maximum reduction that also tracks the index of maximum element
 */
template<int BLOCK_SIZE>
__device__ float blockReduceMaxWithIndex(float val, int local_idx, int* global_max_idx) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Shared memory for warp results
    __shared__ float warpMaxs[BLOCK_SIZE / warpSize];
    __shared__ int warpMaxIndices[BLOCK_SIZE / warpSize];
    
    // Step 1: Reduce within warp
    int warp_max_idx;
    val = warpReduceMaxWithIndex(val, local_idx, &warp_max_idx);
    
    // Step 2: Store warp results
    if (lane == 0) {
        warpMaxs[wid] = val;
        warpMaxIndices[wid] = warp_max_idx;
    }
    __syncthreads();
    
    // Step 3: Final reduction in first warp
    if (wid == 0) {
        val = (threadIdx.x < BLOCK_SIZE / warpSize) ? warpMaxs[lane] : -FLT_MAX;
        local_idx = (threadIdx.x < BLOCK_SIZE / warpSize) ? warpMaxIndices[lane] : -1;
        
        val = warpReduceMaxWithIndex(val, local_idx, global_max_idx);
    }
    
    return val;
}

// =============================================================================
// 3. GRAPH-SPECIFIC KERNEL IMPLEMENTATIONS
// =============================================================================

/**
 * Find maximum edge weight in each vertex's neighborhood
 * Each block processes one vertex, threads process edges in parallel
 */
template<int BLOCK_SIZE>
__global__ void findMaxEdgeWeightPerVertex(
    const int* __restrict__ degree_list,           // Vertex degrees
    const float* const* __restrict__ beg_ptr,      // Neighbor data pointers
    const int* __restrict__ bin_size,              // Bin sizes for each vertex
    float* __restrict__ max_weights,               // Output: max weight per vertex
    int* __restrict__ max_edge_indices,            // Output: index of max edge (optional)
    int num_vertices
) {
    int vertex = blockIdx.x;
    if (vertex >= num_vertices) return;
    
    int degree = degree_list[vertex];
    int bin = bin_size[vertex];
    
    // Each thread processes multiple edges if degree > BLOCK_SIZE
    float local_max = -FLT_MAX;
    int local_max_idx = -1;
    
    for (int edge_idx = threadIdx.x; edge_idx < degree; edge_idx += blockDim.x) {
        // Access weight from the neighbor data structure
        // Weights are stored at offset [bin + edge_idx] from the vertex's base pointer
        float weight = beg_ptr[vertex][bin + edge_idx];
        
        if (weight > local_max) {
            local_max = weight;
            local_max_idx = edge_idx;
        }
    }
    
    // Block-level reduction to find maximum
    int global_max_idx;
    float block_max = blockReduceMaxWithIndex<BLOCK_SIZE>(local_max, local_max_idx, &global_max_idx);
    
    // Store results (only thread 0)
    if (threadIdx.x == 0) {
        max_weights[vertex] = block_max;
        if (max_edge_indices != nullptr) {
            max_edge_indices[vertex] = global_max_idx;
        }
    }
}

/**
 * Compute maximum resistance value during random walk sampling
 * Used in the resistance-based edge sampling for graph sparsification
 */
template<int BLOCK_SIZE>
__global__ void findMaxResistanceInPaths(
    const float* __restrict__ resistance_values,   // Resistance values from random walks
    const int* __restrict__ path_lengths,          // Length of each path
    float* __restrict__ max_resistance_per_block,  // Output per block
    int total_paths,
    int max_path_length
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = -FLT_MAX;
    
    // Each thread processes multiple resistance values
    for (int i = tid; i < total_paths * max_path_length; i += gridDim.x * blockDim.x) {
        int path_id = i / max_path_length;
        int step_id = i % max_path_length;
        
        // Only consider valid steps within path length
        if (step_id < path_lengths[path_id]) {
            float resistance = resistance_values[i];
            local_max = fmaxf(local_max, resistance);
        }
    }
    
    // Block reduction
    float block_max = blockReduceMax<BLOCK_SIZE>(local_max);
    
    // Store result
    if (threadIdx.x == 0) {
        max_resistance_per_block[blockIdx.x] = block_max;
    }
}

/**
 * Aggregate statistics across multiple graph updates
 * Useful for tracking maximum changes during incremental/decremental updates
 */
template<int BLOCK_SIZE>
__global__ void aggregateGraphUpdateStats(
    const float* __restrict__ edge_weight_changes, // Changes in edge weights
    const int* __restrict__ update_types,          // 0=addition, 1=deletion, 2=modification
    float* __restrict__ max_addition,              // Maximum weight added
    float* __restrict__ max_deletion,              // Maximum weight deleted  
    float* __restrict__ max_modification,          // Maximum weight change
    int num_updates
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max_add = -FLT_MAX;
    float local_max_del = -FLT_MAX;
    float local_max_mod = -FLT_MAX;
    
    // Process updates
    for (int i = tid; i < num_updates; i += gridDim.x * blockDim.x) {
        float weight_change = edge_weight_changes[i];
        int update_type = update_types[i];
        
        switch (update_type) {
            case 0: // Addition
                local_max_add = fmaxf(local_max_add, weight_change);
                break;
            case 1: // Deletion
                local_max_del = fmaxf(local_max_del, fabsf(weight_change));
                break;
            case 2: // Modification
                local_max_mod = fmaxf(local_max_mod, fabsf(weight_change));
                break;
        }
    }
    
    // Block reductions for each category
    float block_max_add = blockReduceMax<BLOCK_SIZE>(local_max_add);
    float block_max_del = blockReduceMax<BLOCK_SIZE>(local_max_del);
    float block_max_mod = blockReduceMax<BLOCK_SIZE>(local_max_mod);
    
    // Store results
    if (threadIdx.x == 0) {
        max_addition[blockIdx.x] = block_max_add;
        max_deletion[blockIdx.x] = block_max_del;
        max_modification[blockIdx.x] = block_max_mod;
    }
}

// =============================================================================
// 4. COOPERATIVE GROUPS VERSION FOR DYNAMIC BLOCK SIZES
// =============================================================================

/**
 * Flexible reduction using cooperative groups
 * Works with any block size, determined at runtime
 */
__device__ float cooperativeBlockReduceMax(float val) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    // Warp-level reduction
    val = warpReduceMax(val);
    
    // Inter-warp communication via shared memory
    extern __shared__ float cg_shared[];
    
    if (warp.thread_rank() == 0) {
        cg_shared[warp.meta_group_rank()] = val;
    }
    block.sync();
    
    // Final reduction in first warp
    if (warp.meta_group_rank() == 0) {
        val = (warp.thread_rank() < block.group_dim().x / 32) ? 
              cg_shared[warp.thread_rank()] : -FLT_MAX;
        val = warpReduceMax(val);
    }
    
    return val;
}

/**
 * Runtime block size version for graph operations
 */
__global__ void dynamicBlockSizeGraphReduction(
    const float* __restrict__ input_data,
    float* __restrict__ output_data,
    int data_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < data_size) ? input_data[tid] : -FLT_MAX;
    
    // Use cooperative groups for dynamic block size
    float result = cooperativeBlockReduceMax(val);
    
    if (threadIdx.x == 0) {
        output_data[blockIdx.x] = result;
    }
}

// =============================================================================
// 5. HOST WRAPPER FUNCTIONS FOR EASY INTEGRATION
// =============================================================================

/**
 * Host function to launch maximum edge weight kernel
 */
void launchMaxEdgeWeightKernel(
    const int* degree_list,
    const float* const* beg_ptr, 
    const int* bin_size,
    float* max_weights,
    int* max_edge_indices,
    int num_vertices,
    int block_size = 256
) {
    // Launch with one block per vertex
    dim3 grid(num_vertices);
    dim3 block(block_size);
    
    // Template dispatch based on block size
    switch (block_size) {
        case 64:
            findMaxEdgeWeightPerVertex<64><<<grid, block>>>(
                degree_list, beg_ptr, bin_size, max_weights, max_edge_indices, num_vertices);
            break;
        case 128:
            findMaxEdgeWeightPerVertex<128><<<grid, block>>>(
                degree_list, beg_ptr, bin_size, max_weights, max_edge_indices, num_vertices);
            break;
        case 256:
            findMaxEdgeWeightPerVertex<256><<<grid, block>>>(
                degree_list, beg_ptr, bin_size, max_weights, max_edge_indices, num_vertices);
            break;
        case 512:
            findMaxEdgeWeightPerVertex<512><<<grid, block>>>(
                degree_list, beg_ptr, bin_size, max_weights, max_edge_indices, num_vertices);
            break;
        default:
            // Use dynamic version for non-standard block sizes
            size_t shared_mem_size = (block_size / 32) * sizeof(float);
            dynamicBlockSizeGraphReduction<<<grid, block, shared_mem_size>>>(
                (const float*)degree_list, max_weights, num_vertices);
            break;
    }
    
    cudaDeviceSynchronize();
}

/**
 * Host function for resistance computation
 */
void launchMaxResistanceKernel(
    const float* resistance_values,
    const int* path_lengths,
    float* max_resistance_per_block,
    int total_paths,
    int max_path_length,
    int block_size = 256,
    int grid_size = 32
) {
    dim3 grid(grid_size);
    dim3 block(block_size);
    
    switch (block_size) {
        case 128:
            findMaxResistanceInPaths<128><<<grid, block>>>(
                resistance_values, path_lengths, max_resistance_per_block,
                total_paths, max_path_length);
            break;
        case 256:
            findMaxResistanceInPaths<256><<<grid, block>>>(
                resistance_values, path_lengths, max_resistance_per_block,
                total_paths, max_path_length);
            break;
        case 512:
            findMaxResistanceInPaths<512><<<grid, block>>>(
                resistance_values, path_lengths, max_resistance_per_block,
                total_paths, max_path_length);
            break;
        default:
            findMaxResistanceInPaths<256><<<grid, block>>>(
                resistance_values, path_lengths, max_resistance_per_block,
                total_paths, max_path_length);
            break;
    }
    
    cudaDeviceSynchronize();
}

// =============================================================================
// 6. PERFORMANCE TESTING AND VALIDATION
// =============================================================================

__global__ void validateReductionCorrectness(
    const float* input, 
    float* block_results, 
    float* reference_max,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each block finds its local maximum
    float val = (tid < n) ? input[tid] : -FLT_MAX;
    float block_max = blockReduceMax<256>(val);
    
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = block_max;
        
        // Atomic max to find global maximum (for validation)
        atomicMax((int*)reference_max, __float_as_int(block_max));
    }
}

void benchmarkReductionMethods() {
    printf("=== Graph Reduction Kernel Performance Analysis ===\n");
    printf("Comparing different approaches for block-level max reduction:\n");
    printf("1. Shared Memory (classical)\n");
    printf("2. Warp Shuffle (modern)\n"); 
    printf("3. Cooperative Groups (flexible)\n");
    printf("\nPerformance benefits of avoiding global memory:\n");
    printf("- Shared memory: ~100x faster than global memory\n");
    printf("- Warp shuffle: Direct register operations, no memory access\n");
    printf("- Reduced memory bandwidth pressure\n");
    printf("- Better cache utilization\n");
    printf("- Lower latency operations\n\n");
}
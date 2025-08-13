#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>

using namespace cooperative_groups;

// =============================================================================
// 1. SHARED MEMORY REDUCTION (Classical Approach)
// =============================================================================

template<int BLOCK_SIZE>
__device__ float blockReduceMax_SharedMem(float val) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    
    shared[tid] = val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    
    return shared[0];
}

// =============================================================================
// 2. WARP SHUFFLE REDUCTION (Modern, Efficient)
// =============================================================================

__device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ float blockReduceMax_WarpShuffle(float val) {
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE too large");
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Reduce within warp using shuffle
    val = warpReduceMax(val);
    
    // Write reduced value to shared memory
    __shared__ float warpMaxs[BLOCK_SIZE / warpSize];
    if (lane == 0) warpMaxs[wid] = val;
    __syncthreads();
    
    // Read from shared memory only if that warp existed
    val = (threadIdx.x < BLOCK_SIZE / warpSize) ? warpMaxs[lane] : -FLT_MAX;
    
    // Final reduce within first warp
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}

// =============================================================================
// 3. COOPERATIVE GROUPS REDUCTION (Most Flexible)
// =============================================================================

template<typename Group>
__device__ float cooperativeReduceMax(Group g, float val) {
    int lane = g.thread_rank();
    
    // Reduce within group using shuffle
    for (int i = g.size() / 2; i > 0; i /= 2) {
        float other = g.shfl_down(val, i);
        val = fmaxf(val, other);
    }
    
    return val;
}

__device__ float blockReduceMax_CooperativeGroups(float val) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    // Reduce within warp
    val = cooperativeReduceMax(warp, val);
    
    // Shared memory for inter-warp communication
    __shared__ float warp_maxs[32]; // Max 1024 threads = 32 warps
    
    if (warp.thread_rank() == 0) {
        warp_maxs[warp.meta_group_rank()] = val;
    }
    block.sync();
    
    // Create a group of warp leaders
    if (warp.meta_group_rank() == 0) {
        val = (warp.thread_rank() < block.group_dim().x / 32) ? 
              warp_maxs[warp.thread_rank()] : -FLT_MAX;
        val = cooperativeReduceMax(warp, val);
    }
    
    return val;
}

// =============================================================================
// 4. TEMPLATE-BASED APPROACH FOR DIFFERENT BLOCK SIZES
// =============================================================================

template<int BLOCK_SIZE>
struct BlockReduceMax {
    __device__ static float reduce(float val) {
        if constexpr (BLOCK_SIZE <= 32) {
            // Use warp shuffle for small blocks
            return warpReduceMax(val);
        } else {
            // Use hybrid approach for larger blocks
            return blockReduceMax_WarpShuffle<BLOCK_SIZE>(val);
        }
    }
};

// Specializations for common block sizes
template<>
struct BlockReduceMax<32> {
    __device__ static float reduce(float val) {
        return warpReduceMax(val);
    }
};

template<>
struct BlockReduceMax<64> {
    __device__ static float reduce(float val) {
        return blockReduceMax_WarpShuffle<64>(val);
    }
};

template<>
struct BlockReduceMax<256> {
    __device__ static float reduce(float val) {
        return blockReduceMax_WarpShuffle<256>(val);
    }
};

// =============================================================================
// 5. KERNEL EXAMPLES USING THE REDUCTIONS
// =============================================================================

// Example: Find maximum in array segments (one block per segment)
template<int BLOCK_SIZE>
__global__ void findMaxPerBlock_Kernel(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data (with bounds checking)
    float val = (tid < n) ? input[tid] : -FLT_MAX;
    
    // Perform block reduction
    float block_max = BlockReduceMax<BLOCK_SIZE>::reduce(val);
    
    // Write result (only thread 0 in each block)
    if (threadIdx.x == 0) {
        output[blockIdx.x] = block_max;
    }
}

// Example: Graph sparsification - find maximum edge weight in neighborhood
template<int BLOCK_SIZE>
__global__ void findMaxEdgeWeight_Kernel(
    int* row_ptr,           // CSR row pointers
    int* col_indices,       // CSR column indices  
    float* edge_weights,    // Edge weights
    float* max_weights,     // Output: max weight per vertex
    int num_vertices
) {
    int vertex = blockIdx.x;
    if (vertex >= num_vertices) return;
    
    int start = row_ptr[vertex];
    int end = row_ptr[vertex + 1];
    int degree = end - start;
    
    float local_max = -FLT_MAX;
    
    // Each thread processes multiple edges if degree > BLOCK_SIZE
    for (int i = threadIdx.x; i < degree; i += blockDim.x) {
        if (start + i < end) {
            float weight = edge_weights[start + i];
            local_max = fmaxf(local_max, weight);
        }
    }
    
    // Reduce within block
    float block_max = BlockReduceMax<BLOCK_SIZE>::reduce(local_max);
    
    // Store result
    if (threadIdx.x == 0) {
        max_weights[vertex] = block_max;
    }
}

// =============================================================================
// 6. DYNAMIC BLOCK SIZE VERSION (Runtime Block Size)
// =============================================================================

__device__ float dynamicBlockReduceMax(float val, int block_size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    sdata[tid] = val;
    __syncthreads();
    
    // Dynamic reduction
    for (int stride = 1; stride < block_size; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < block_size) {
            sdata[index] = fmaxf(sdata[index], sdata[index + stride]);
        }
        __syncthreads();
    }
    
    return sdata[0];
}

// =============================================================================
// 7. BENCHMARK/TEST FUNCTIONS
// =============================================================================

__global__ void benchmark_SharedMem(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? input[tid] : -FLT_MAX;
    
    float result = blockReduceMax_SharedMem<256>(val);
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = result;
    }
}

__global__ void benchmark_WarpShuffle(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? input[tid] : -FLT_MAX;
    
    float result = blockReduceMax_WarpShuffle<256>(val);
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = result;
    }
}

__global__ void benchmark_CooperativeGroups(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? input[tid] : -FLT_MAX;
    
    float result = blockReduceMax_CooperativeGroups(val);
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = result;
    }
}

// =============================================================================
// 8. HOST CODE FOR TESTING
// =============================================================================

void test_reductions() {
    const int n = 1024 * 1024;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    
    // Allocate host memory
    float* h_input = new float[n];
    float* h_output = new float[num_blocks];
    
    // Initialize with random data
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, num_blocks * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test different approaches
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark shared memory approach
    cudaEventRecord(start);
    benchmark_SharedMem<<<num_blocks, block_size>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);
    printf("Shared Memory Reduction: %.3f ms\n", time_shared);
    
    // Benchmark warp shuffle approach
    cudaEventRecord(start);
    benchmark_WarpShuffle<<<num_blocks, block_size>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_shuffle;
    cudaEventElapsedTime(&time_shuffle, start, stop);
    printf("Warp Shuffle Reduction: %.3f ms\n", time_shuffle);
    
    // Benchmark cooperative groups approach
    cudaEventRecord(start);
    benchmark_CooperativeGroups<<<num_blocks, block_size>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_coop;
    cudaEventElapsedTime(&time_coop, start, stop);
    printf("Cooperative Groups Reduction: %.3f ms\n", time_coop);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_reductions();
    return 0;
}
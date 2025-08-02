/**
 * GPU Graph Data Structures for dyGRASS Random Walk Incremental Processing
 * 
 * This header defines CUDA-optimized data structures for:
 * - GPU-resident graph storage and manipulation
 * - Random walk sampling and path finding
 * - Dynamic graph updates (adding edges incrementally)
 * - Memory management between host and device
 * 
 * Key Components:
 * - gpu_graph: Main GPU graph structure with CSR format
 * - Random_walk_data: CUDA kernel data for random walk computation
 * - Error handling macros for CUDA operations
 * 
 * Updated for dyGRASS incremental graph sparsification
 */

#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include <iostream>
// #include <curand.h>
#include "header.h"
// #include "util.h"
#include "herror.h"
// #include "graph.h"
#include "functions.h"
#include <curand_kernel.h>
#include <assert.h>

using namespace std;

/**
 * CUDA Error Handling Utilities
 * 
 * Provides centralized error checking for all CUDA API calls
 * to ensure failures are caught and reported immediately.
 */
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", \
        cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// Macro for convenient error checking: H_ERR(cudaMalloc(...))
#define H_ERR( err )(HandleError( err, __FILE__, __LINE__ ))


/**
 * GPU-Optimized Graph Data Structure for Incremental Random Walk Sampling
 * 
 * This class manages a graph stored in GPU memory using Compressed Sparse Row (CSR) format
 * optimized for random walk operations. It supports dynamic edge additions during 
 * incremental graph sparsification experiments.
 * 
 * Memory Architecture:
 * - Dual storage: Host (CPU) and Device (GPU) copies
 * - CSR format with neighbor arrays organized for coalesced GPU access
 * - Dynamic expansion area for new edges added during sampling
 * 
 * Key Features:
 * - Batched sample processing (processes samples in groups of ~sample_count/10)
 * - Incremental edge addition when random walks find no connecting paths
 * - Specialized memory layout for GPU-efficient neighbor access
 */
class gpu_graph
{
	public:
        // === Graph Size Information ===
		index_t vert_count;          // Number of vertices in graph
		index_t edge_count;          // Number of edges in original graph
        index_t sample_total_count;  // Total sample edges to process
		index_t sample_count;        // Current batch size (~total/10)
        index_t no_path_count;       // Count of edges added (no path found)
        index_t sample_ptr;          // Current position in sample array
        index_t ext_size;            // Number of edges added to extension array

        // === Host (CPU) Memory Pointers ===
        weight_t * mtx;              // Original edge matrix [src, dest, weight] triplets
        weight_t * ext;              // Extension edges added during processing
        weight_t ** beg_ptr;         // Pointers to start of each vertex's neighbor list
        weight_t * neighbors_data;   // Packed neighbor data: [adj|weights|from|reverse]
        vertex_t * degree_list;      // Vertex degrees (updated as edges are added)
        vertex_t * sample_nodes_all; // All sample edge endpoints to process
        weight_t * sample_weights;   // Weights for sample edges
        weight_t * extra_neighbors_data; // Space for dynamically added neighbors
        

        // === Device (GPU) Memory Pointers ===
        weight_t ** beg_ptr_device;         // GPU copy of neighbor list pointers
        weight_t * neighbors_data_device;   // GPU copy of packed neighbor data
        vertex_t * degree_list_device;      // GPU copy of vertex degrees
        vertex_t * sample_nodes_device;     // GPU copy of current sample batch
        weight_t * sample_weights_device;   // GPU copy of current sample weights
        weight_t * extra_neighbors_data_device; // GPU space for new neighbors

        // === Host-Generated Device Pointers ===
        // These are computed on host but used to update device pointer arrays
        weight_t ** beg_ptr_device_content; // Host-computed device addresses
		unsigned  extra_offset;             // Offset in extra neighbor space

        // === Algorithm Parameters ===
        float distortion;                   // Distortion threshold for random walks

        

	public:
        /**
         * Constructor: Initialize GPU graph from CPU CSR graph
         * 
         * Converts a CPU-based CSRGraph to GPU-optimized format with:
         * - Specialized neighbor data layout for GPU coalescing
         * - Batch processing setup (divide samples into ~10 batches)
         * - Memory allocation for dynamic edge additions
         * 
         * @param ginst Input CSR graph from CPU processing
         * @param distortion Random walk distortion threshold parameter
         */
        gpu_graph(
			CSRGraph ginst,
            float distortion
        )
		{   
            this->distortion = distortion;

            // Copy basic graph dimensions
            this->vert_count = ginst.vert_count;
			this->edge_count = ginst.edge_count;
            this->sample_total_count = ginst.sample_count;
            // Divide samples into batches (~10 batches for memory management)
            this->sample_count = (ginst.sample_count + 10 -1)/10;
            
            // Initialize counters
            this->no_path_count = 0;     // No edges added yet
            this->extra_offset = 0;      // No extra space used yet
            this->sample_ptr = 0;        // Start at first sample

            // Convert original edge matrix to flat triplet format
            auto [mtx, ext] = createMtx(ginst.mtx);
            this->mtx = mtx;
            this->ext = ext;            // Extension array for new edges
            this->ext_size = 0;         // No extensions yet

            // Create GPU-optimized neighbor arrays on host
            auto [beg_ptr, neighbors_data] = createNeighborArray(ginst);
            this->beg_ptr = beg_ptr;
            this->neighbors_data = neighbors_data;

            // Reference original sample data (not copied, just referenced)
            this->degree_list = ginst.degree_list;
            this->sample_nodes_all = ginst.sample_nodes;
            this->sample_weights = ginst.sample_weights;

            // Allocate space for device pointer computations
            this->beg_ptr_device_content = new weight_t*[vert_count];
            // Extra space starts after main neighbor data
            this->extra_neighbors_data = neighbors_data + edge_count*4;
        }

        /**
         * Convert edge tuple vector to flat array format
         * 
         * Transforms vector<tuple<src,dest,weight>> to flat array format:
         * [src1, dest1, weight1, src2, dest2, weight2, ...]
         * This provides better memory locality for GPU processing.
         * 
         * @param mtx Vector of edge tuples from CSR graph
         * @return Pair of arrays: (original_edges, extension_space)
         */
        tuple<weight_t*, weight_t*> createMtx(vector<tuple<vertex_t, vertex_t, weight_t>> mtx){
            int len = mtx.size();
            weight_t * mtx_data = new weight_t[len*3];  // Flat array: 3 values per edge

            // Convert tuples to flat triplet format
            for (int i = 0; i < len; i += 1){
                mtx_data[i*3] = get<0>(mtx[i]);      // Source vertex
                mtx_data[i*3 + 1] = get<1>(mtx[i]);  // Destination vertex
                mtx_data[i*3 + 2] = get<2>(mtx[i]);  // Edge weight
            }

            // Allocate space for edges that will be added during processing
            weight_t * ext_data = new weight_t[sample_total_count*3];
            return {mtx_data, ext_data};
        }

        /**
         * Create GPU-optimized neighbor arrays from CSR graph
         * 
         * Transforms standard CSR format into GPU-friendly layout:
         * For each vertex with degree D, stores neighbors as:
         * [neighbor1, neighbor2, ..., neighborD,
         *  weight1, weight2, ..., weightD,
         *  from1, from2, ..., fromD,
         *  reverse1, reverse2, ..., reverseD]
         * 
         * This layout enables coalesced memory access in GPU kernels.
         * 
         * @param graph Input CSR graph structure
         * @return Pair: (pointer_array, packed_neighbor_data)
         */
        tuple<weight_t**, weight_t*> createNeighborArray(const CSRGraph& graph) {
            // Array of pointers to each vertex's neighbor data
            weight_t** beg_ptr = new weight_t*[graph.vert_count];

            // Packed neighbor data: factor of 8 = 4 arrays * 2 (original + extra space)
            weight_t* neighbors_data = new weight_t[graph.edge_count * 4 * 2];
            size_t offset = 0;
            
            for (size_t i = 0; i < graph.vert_count; i++) {
                // Point to start of this vertex's neighbor block
                beg_ptr[i] = neighbors_data + offset;
                
                // Copy neighbor data in blocked format for vertex i
                index_t degree = graph.degree[i];
                for (size_t j = graph.begin[i]; j < graph.begin[i + 1]; j++) {
                    // Block 1: Neighbor vertex IDs
                    neighbors_data[offset] = static_cast<weight_t>(graph.adj[j]);
                    // Block 2: Edge weights (offset by degree)
                    neighbors_data[offset + degree] = graph.weight[j];
                    // Block 3: Original edge indices (offset by 2*degree)
                    neighbors_data[offset + 2 * degree] = static_cast<weight_t>(graph.from[j]);
                    // Block 4: Reverse edge indices (offset by 3*degree)
                    neighbors_data[offset + 3 * degree] = static_cast<weight_t>(graph.reverse[j]);
                    offset++;
                }
                // Move offset past all 4 blocks for this vertex
                offset += 3 * degree;
            }

            return {beg_ptr, neighbors_data};
        }

        /**
         * Transfer graph data from host to GPU device memory
         * 
         * Performs initial GPU memory allocation and data transfer:
         * 1. Allocate GPU memory for all graph structures
         * 2. Compute device pointer addresses for neighbor arrays
         * 3. Copy all graph data to GPU
         * 4. Copy first batch of sample data
         */
        void copyToDevice(){
            // === GPU Memory Allocation ===
            HRR(cudaMalloc((void ***)&beg_ptr_device, sizeof(weight_t*)*vert_count)); 
            HRR(cudaMalloc((void **)&neighbors_data_device, sizeof(weight_t)*edge_count*4 * 2));
            HRR(cudaMalloc((void **)&degree_list_device, sizeof(vertex_t)*vert_count));
            HRR(cudaMalloc((void **)&sample_nodes_device, sizeof(vertex_t)*sample_count*2));
            HRR(cudaMalloc((void **)&sample_weights_device, sizeof(weight_t)*sample_count));
            
            // === Compute Device Pointer Addresses ===
            // Calculate where each vertex's neighbor data will be located on GPU
            unsigned offset = 0;
            for (int i = 0; i < vert_count; i++){
                weight_t * ptr = neighbors_data_device + offset;
                beg_ptr_device_content[i] = reinterpret_cast<weight_t*>(ptr);
                offset += degree_list[i]*4;  // Each vertex uses 4*degree space
            }
            
            // Extra space for dynamically added edges starts after main data
            extra_neighbors_data_device = neighbors_data_device + offset;
            assert(extra_neighbors_data_device == neighbors_data_device + edge_count*4);
            
            // === Copy Data to GPU ===
            HRR(cudaMemcpy(neighbors_data_device, neighbors_data, sizeof(weight_t)*edge_count*4 * 2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(degree_list_device, degree_list, sizeof(vertex_t)*vert_count, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(beg_ptr_device, beg_ptr_device_content, sizeof(weight_t*)*vert_count, cudaMemcpyHostToDevice));
            
            // Copy first batch of sample data
            HRR(cudaMemcpy(sample_nodes_device, (sample_nodes_all+sample_ptr), sizeof(vertex_t)*sample_count*2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sample_weights_device, (sample_weights + sample_ptr/2), sizeof(weight_t)*sample_count, cudaMemcpyHostToDevice));
        }

        /**
         * Process random walk results and update graph incrementally
         * 
         * Core incremental sparsification logic:
         * 1. For each sample edge, check if random walk found a path
         * 2. If NO path found: add the sample edge directly to the graph
         * 3. If path found: the edge is "sparsified" (represented by the path)
         * 4. Update graph structure with new edges and sync to GPU
         * 
         * This implements the key dyGRASS principle: only add edges when
         * random walks cannot find alternative paths.
         * 
         * @param path_selected_flag Array indicating if path was found for each sample
         * @param final_conductance Array of conductance values for found paths
         */
        void updateGraphFromResult(
            int* path_selected_flag, 
            float* final_conductance
        ){
            // Process results for current batch of samples
            for (int i = 0; i < sample_count; i++){
                int find_at = path_selected_flag[i]; // -1 if no path found
                weight_t weight = sample_weights[sample_ptr/2 + i];
                
                // === Path Found: Edge is Successfully Sparsified ===
                if(find_at != -1){ 
                    // Do nothing - the sample edge is represented by the found path
                    // This is the core sparsification: avoid adding redundant edges
                }
                // === No Path Found: Must Add Edge to Maintain Connectivity ===
                else{ 
                    no_path_count += 1;

                    // Get source and destination from sample array layout
                    int source = sample_nodes_all[sample_ptr + i];
                    int target = sample_nodes_all[sample_ptr + sample_count + i];
                    
                    // Add edge to extension array in triplet format
                    ext[ext_size*3] = source;
                    ext[ext_size*3 + 1] = target;
                    ext[ext_size*3 + 2] = weight;

                    // Update vertex degrees (increment before use)
                    int degree_source = ++degree_list[source];
                    int degree_target = ++degree_list[target];

                    // Add bidirectional edges to neighbor arrays
                    manageExtraNeighborsData(source, target, weight, degree_source, degree_target);
                    manageExtraNeighborsData(target, source, weight, degree_target, degree_source);

                    ext_size += 1;
                }
            }
            cout << "no path count: " << no_path_count << endl;

            // === Sync Updated Graph Back to GPU ===
            HRR(cudaMemcpy(neighbors_data_device, neighbors_data, sizeof(weight_t)*edge_count*4 * 2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(beg_ptr_device, beg_ptr_device_content, sizeof(weight_t*)*vert_count, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(degree_list_device, degree_list, sizeof(vertex_t)*vert_count, cudaMemcpyHostToDevice));
        }

        /**
         * Add new neighbor to vertex's neighbor array with dynamic memory management
         * 
         * When adding edges dynamically, vertex neighbor arrays must be expanded.
         * This function:
         * 1. Copies existing neighbor data to new larger location
         * 2. Adds the new neighbor data at the end
         * 3. Updates pointers to point to new location
         * 
         * Memory layout for vertex with degree D:
         * [neighbors: N1,N2,...,ND] [weights: W1,W2,...,WD] 
         * [from: F1,F2,...,FD] [reverse: R1,R2,...,RD]
         * 
         * @param node Vertex getting new neighbor
         * @param neighbor New neighbor vertex ID
         * @param weight Weight of new edge
         * @param new_degree Updated degree after adding edge
         * @param reverse_index Index for reverse edge lookup
         */
        void manageExtraNeighborsData(int node, int neighbor, weight_t weight, int new_degree, int reverse_index){
            weight_t * ptr_old = beg_ptr[node];        // Current neighbor data location
            weight_t * ptr_new = extra_neighbors_data + extra_offset; // New location in extra space

            // Copy existing neighbor data to new location with expanded layout
            for (int i = 0, j = 0; i < (new_degree - 1) * 4; i++, j++){
                if (j % new_degree == new_degree - 1){ // Skip slot for new neighbor
                    j++;
                }
                ptr_new[j] = ptr_old[i];
            }
            
            // Add new neighbor data in the reserved slots
            ptr_new[new_degree - 1] = static_cast<weight_t>(neighbor);  // Neighbor ID
            ptr_new[2 * new_degree - 1] = weight;                      // Edge weight
            ptr_new[3 * new_degree - 1] = static_cast<weight_t>(edge_count/2 + ext_size); // From index
            ptr_new[4 * new_degree - 1] = static_cast<weight_t>(reverse_index - 1);       // Reverse index
            
            // Update host pointer to new location
            beg_ptr[node] = ptr_new;

            // Update device pointer mapping
            weight_t * ptr_new_device = reinterpret_cast<weight_t*>(extra_neighbors_data_device + extra_offset);
            beg_ptr_device_content[node] = ptr_new_device;
            
            // Advance offset for next allocation
            extra_offset += new_degree * 4;
        }
            

        /**
         * Advance to next batch of sample edges for processing
         * 
         * Implements batched processing by loading the next segment of sample edges.
         * Handles final batch size adjustment when approaching end of sample array.
         * 
         * @return true if more batches remain, false if this is the final batch
         */
        bool updateSampleEdges(){
            bool not_empty = true;
            
            // Move to next batch (factor of 2 because samples stored as [src1,src2,...,dest1,dest2,...])
            sample_ptr += sample_count * 2;
            
            // Check if we're approaching the end of sample array
            if (sample_ptr + sample_count * 2 >= sample_total_count * 2){
                // Adjust batch size for final partial batch
                sample_count = (sample_total_count * 2 - sample_ptr) / 2;
                not_empty = false; // This will be the final batch
            }

            // Copy next batch of sample data to GPU
            HRR(cudaMemcpy(sample_nodes_device, (sample_nodes_all+sample_ptr), sizeof(vertex_t)*sample_count*2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sample_weights_device, (sample_weights + sample_ptr/2), sizeof(weight_t)*sample_count, cudaMemcpyHostToDevice));

            return not_empty;
        }


        /**
         * Save current graph (original + added edges) to MTX format file
         * 
         * Outputs the final sparsified graph containing:
         * 1. All original edges that were not sparsified
         * 2. All edges added during incremental processing (no path found)
         * 
         * NOTE: Output uses 1-based indexing for compatibility with MTX format!
         * 
         * @param file_name Output file path for the sparsified graph
         */
        void saveCurrentMtx(string file_name){
            std::cout << "saving mtx to " << file_name << std::endl;
            std::ofstream txt_file(file_name, std::ios::out);
            if (txt_file.is_open()){
                
                // Write original edges (convert 0-based to 1-based indexing)
                for (int i = 0; i < edge_count/2; i++){
                    txt_file << 
                    static_cast<int>(mtx[i*3]) + 1 << " " <<      // Source + 1
                    static_cast<int>(mtx[i*3 + 1]) + 1 << " " <<  // Destination + 1
                    mtx[i*3 + 2] << endl;                         // Weight (unchanged)
                }

                // Write extension edges added during processing
                for (int i = 0; i < ext_size; i++){
                    txt_file << 
                    static_cast<int>(ext[i*3]) + 1 << " " <<      // Source + 1
                    static_cast<int>(ext[i*3 + 1]) + 1 << " " <<  // Destination + 1
                    ext[i*3 + 2] << endl;                         // Weight (unchanged)
                }

                txt_file.close();
            }
            else{
                std::cerr << "Unable to open file"<< endl;
            }
            cout << "Updated graph saved to " << file_name << endl;
        }

        ~gpu_graph(){
            cout << "gpu_graph" << endl;
            delete[] beg_ptr;
            delete[] neighbors_data;
            // delete[] degree_list; // don't delete this, it is from the graph
            // delete[] extra_neighbors_data;
            delete[] beg_ptr_device_content;
            delete ext;
            delete mtx;

            cudaFree(beg_ptr_device);
            cudaFree(neighbors_data_device);
            cudaFree(degree_list_device);
            cudaFree(sample_nodes_device);
            cudaFree(sample_weights_device);
            // cudaFree(extra_neighbors_data_device);
            cout << "gpu_graph deleted" << endl;
        }

            


};

/**
 * CUDA Data Structure for Random Walk Computation
 * 
 * Manages GPU memory and computation state for parallel random walk execution.
 * Each CUDA thread processes one sample edge, attempting to find connecting paths
 * through random walks on the graph.
 * 
 * Key Components:
 * - Random number generation state (cuRAND)
 * - Path finding results and flags
 * - Conductance computation for path quality evaluation
 * - Shared memory arrays for GPU reduction operations
 */
class Random_walk_data{

    public:
    int sample_num;                       // Number of sample edges to process per batch
    int n_steps;                          // Maximum steps per random walk
    
    // === Path Finding Results ===
    // int* path_selected;                // [UNUSED] Full path data
	// int* path_selected_device;         // [UNUSED] GPU copy
	int* path_selected_flag;              // Host: -1 if no path found, step count if found
	int* path_selected_flag_device;       // GPU copy of path flags
    
    // === Conductance Computation Arrays ===
    float* conductance_shared_mem;        // GPU shared memory for per-thread conductance values
	int* conductance_index_shared_mem;    // GPU shared memory for thread indices (reduction)
    float* final_conductance;             // Host: final conductance for each successful path
    float* final_conductance_device;      // GPU copy of final conductance values

    // === Random Number Generation ===
    curandState *global_state;            // GPU: per-thread random number generator state
    
    /**
     * Constructor: Allocate memory for random walk computation
     * 
     * Sets up host and device memory for parallel random walk execution:
     * - Host arrays for results retrieval
     * - GPU arrays for kernel computation
     * - Random number generator initialization
     * 
     * @param n_samples Number of sample edges per batch
     * @param n_threads Number of CUDA threads (typically matches n_samples)
     */
    Random_walk_data(
        int n_samples, 
        int n_threads 
        ){
        this->sample_num = n_samples;
        
        // === Allocate Host Memory ===
        path_selected_flag = (int *)malloc(sizeof(int)*n_samples);
        final_conductance = (float *)malloc(sizeof(float)*n_samples);

        // === Allocate GPU Memory ===
        HRR(cudaMalloc(&global_state, n_threads * sizeof(curandState)));
        HRR(cudaMalloc((void **) &path_selected_flag_device, sizeof(int)*n_samples));
        HRR(cudaMalloc((void **) &conductance_shared_mem, sizeof(float)*n_threads));
        HRR(cudaMalloc((void **) &conductance_index_shared_mem, sizeof(int)*n_threads));
        HRR(cudaMalloc((void **) &final_conductance_device, sizeof(float)*n_samples));
    };


    /**
     * Copy random walk results from GPU back to host memory
     * 
     * Transfers computation results after CUDA kernel execution:
     * - Path finding flags (success/failure for each sample)
     * - Conductance values for successful paths
     */
    void get_result(){
        HRR(cudaMemcpy(path_selected_flag, path_selected_flag_device, sizeof(int)*sample_num, cudaMemcpyDeviceToHost));
        HRR(cudaMemcpy(final_conductance, final_conductance_device, sizeof(float)*sample_num, cudaMemcpyDeviceToHost));
    }
    
    /**
     * Destructor: Clean up all allocated memory
     * 
     * Properly deallocates both host and device memory to prevent leaks.
     */
    ~Random_walk_data(){
        // Free GPU memory
        HRR(cudaFree(global_state));
        HRR(cudaFree(path_selected_flag_device));
        HRR(cudaFree(conductance_shared_mem));
        HRR(cudaFree(conductance_index_shared_mem));
        HRR(cudaFree(final_conductance_device));
        
        // Free host memory
        free(path_selected_flag);
        free(final_conductance);
        
        cout <<"Random walk data deleted" << endl;
    };

};
#endif

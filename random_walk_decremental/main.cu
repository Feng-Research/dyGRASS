/**
 * dyGRASS Random Walk Decremental Graph Sparsification - Main CUDA Implementation
 * 
 * This program implements decremental graph sparsification using random walk sampling:
 * 
 * Algorithm Overview:
 * 1. Load updated dense graph (with candidate edges removed) and current sparse graph
 * 2. For each candidate edge to remove from sparse graph:
 *    - Remove edge from both dense and sparse graphs
 *    - Launch parallel random walks on updated dense graph to find alternative paths
 *    - If alternative paths exist: keep edge removed (successful sparsification)
 *    - If no paths found: restore edge using heuristic recovery (maintain connectivity)
 * 3. Process candidates in batches to manage GPU memory
 * 4. Output final sparsified graph and added recovery edges
 * 
 * Key Innovation: Dual graph approach allows testing connectivity on updated dense graph
 * while progressively sparsifying the result graph, ensuring connectivity preservation.
 * 
 * CUDA Implementation:
 * - Each CUDA block processes one candidate edge removal
 * - Multiple threads per block perform parallel random walks on updated dense graph
 * - Reduction finds best alternative path (highest conductance) among threads
 * - Heuristic recovery adds short paths for essential connectivity
 */

#include <stdio.h>
#include <string.h>
// #include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <curand.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <chrono>
#include "functions.h"
#include "gpu_graph.cuh"
#include "helper_cuda.h"

// #include "sampler.cuh"

// Maximum steps allowed per random walk (prevents infinite loops)
#define max_steps 100
using namespace std;


/**
 * CUDA Kernel: Parallel Random Walk Path Finding for Decremental Sparsification
 * 
 * This is the core of the decremental dyGRASS algorithm. Each CUDA block processes
 * one candidate edge removal by testing connectivity through random walks.
 * 
 * Thread Organization:
 * - Each block handles one candidate edge removal (src -> dest)
 * - Multiple threads per block perform independent random walks on DENSE graph
 * - Threads use reduction to find the best alternative path (highest conductance)
 * 
 * Decremental Algorithm:
 * 1. Start random walk at source vertex (using updated dense graph structure)
 * 2. Randomly navigate through updated dense graph (avoiding backtracking)
 * 3. Record full path for later reconstruction if target reached
 * 4. Calculate conductance = 1/total_resistance for successful paths
 * 5. Use reduction to find best path among all threads in block
 * 
 * Key Difference from Incremental: Uses updated dense graph for pathfinding while
 * testing edges that were removed from sparse graph.
 * 
 * @param all_data  Random walk computation state and results
 * @param G         GPU dual graph structure with updated dense graph for pathfinding
 * @param n_samples Number of candidate edges in current batch
 * @param n_steps   Maximum steps allowed per random walk
 */
__global__ void
random_walk_decremental(
			Random_walk_data_decremental * all_data,
			GPU_Dual_Graph * G, // pass by value, because data inside already on GPU, only pointer to data is passed
			unsigned int n_samples, 
			int n_steps
			){

	// === Thread and Sample Setup ===
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Initialize per-thread random number generator
	curandState local_state = all_data->global_state[tid];
	curand_init(tid, 10, 7, &local_state);

	// Get candidate edge endpoints from batch layout
	unsigned int sourceIndex = G->decremental_sample_nodes_device[blockIdx.x]; // first half
	unsigned int targetIndex = G->decremental_sample_nodes_device[blockIdx.x+n_samples]; // second half
	
	// === Random Walk State ===
	unsigned int currentVertex = sourceIndex;
	unsigned int previousVertex = 0;
	int targetFoundAt = -1;        // Step number where target was found (-1 if not found)
	int step_count = 1;            // Start counting from source (+1)
	unsigned int path [max_steps]; // Full path storage for reconstruction
	float total_R = 0;             // Total resistance (sum of 1/weight) along path

	// === STEP 1: Initialize Path and Select First Neighbor ===
	// Random walks begin from source vertex using DENSE graph structure
	path[0] = sourceIndex;
	unsigned int degree = G->dense_degree_list_device[currentVertex];      // Current degree (after edge removals)
	unsigned int original_degree = G->dense_degree_original_device[currentVertex]; // Original degree (for memory layout)
	float * offset = G->dense_beg_ptr_device[currentVertex];  // Points to neighbor data block
	unsigned int next = curand(&local_state) % degree;       // Random neighbor selection
	unsigned int nextVertex = static_cast<unsigned int>(offset[next]);
	float weight = offset[next + original_degree];  // Weights stored offset by original_degree
	total_R += 1/weight;  // Accumulate resistance (1/weight)
	path[step_count] = nextVertex;  // Record step in path

	// === STEP 2: Check if Target Reached Immediately ===
	if (nextVertex != targetIndex){
		
		// === STEP 3: Continue Random Walk Until Termination ===
		previousVertex = currentVertex;
		currentVertex = nextVertex;
		step_count++;

		while(step_count < n_steps){

			degree = G->dense_degree_list_device[currentVertex];
			if (degree == 1){
				break; // Reached dead end (leaf node)
			}
			if (degree == 0) {
				// Error condition: isolated vertex
				printf("Error: degree is 0\n");
			}

			// === Neighbor Selection with Non-Backtracking ===
			original_degree = G->dense_degree_original_device[currentVertex];
			offset = G->dense_beg_ptr_device[currentVertex];
			// Select from degree-1 neighbors (exclude returning to previous)
			unsigned int next = curand(&local_state) % (degree - 1);
			nextVertex = static_cast<unsigned int>(offset[next]);

			// If selected neighbor is previous vertex, choose last neighbor instead
			if (nextVertex == previousVertex){
				nextVertex = static_cast<unsigned int>(offset[degree - 1]);
				weight = offset[degree - 1 + original_degree];  // Weight at last position
			}
			else{
				weight = offset[next + original_degree];  // Weight at selected position
			}

			// Record step and accumulate resistance
			path[step_count] = nextVertex;
			total_R += 1/weight;

			// === Termination Conditions ===
			if(nextVertex == targetIndex){
				targetFoundAt = step_count;  // Success! Record step count
				break;
			}

			// Continue walk
			previousVertex = currentVertex;
			currentVertex = nextVertex;
			step_count++;
		}
	}
	else{ 
		// Target found immediately (first neighbor)
		targetFoundAt = step_count; 
	}

	// === STEP 4: Calculate Conductance for Successful Paths ===
	if (targetFoundAt != -1){
		// Path found: conductance = 1/total_resistance (higher is better)
		all_data->conductance_shared_mem[tid] = 1/total_R;
	}else{
		// No path found: mark with negative conductance
		all_data->conductance_shared_mem[tid] = -1;
	}
	__syncthreads();  // Ensure all threads have computed conductance

	// === STEP 5: Parallel Reduction to Find Best Path ===
	// Among all threads in this block, find the one with highest conductance
	// This represents the best alternative path found for this candidate edge
	all_data->conductance_index_shared_mem[tid] = threadIdx.x;
	float left, right;
	
	// Binary tree reduction to find maximum conductance
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			left = all_data->conductance_shared_mem[tid];
			right = all_data->conductance_shared_mem[tid + s];

			// Keep the higher conductance (better path)
			bool isSmall = (left < right);
			all_data->conductance_shared_mem[tid] = isSmall ? right : left;
			// Track which thread had the better path
			all_data->conductance_index_shared_mem[tid] = isSmall ? all_data->conductance_index_shared_mem[tid + s]: all_data->conductance_index_shared_mem[tid];
		}
		__syncthreads();
	}
	
	// === STEP 6: Winner Thread Records Final Result ===
	// Thread with best conductance writes result for this candidate edge
	int max_conductance_thread = all_data->conductance_index_shared_mem[blockDim.x *blockIdx.x]; // get the maximum index from the first thread of the block
	if (threadIdx.x == max_conductance_thread){
		// Record whether alternative path was found (step count or -1)
		all_data->path_selected_flag_device[blockIdx.x] = targetFoundAt;
		
		if (targetFoundAt > 0){ // >0 means the target is found
			// Store the complete alternative path for reconstruction
			for (int i = 0; i <= targetFoundAt; i++){
				all_data->path_selected_device[blockIdx.x * n_steps + i] = path[i];
			}
			// Record the conductance of the best alternative path
			all_data->final_conductance_device[blockIdx.x] = 1/total_R;
		}
	}

}

/**
 * CUDA Kernel: Short Random Walks for Heuristic Recovery
 * 
 * This kernel performs short (3-step) random walks for heuristic connectivity recovery.
 * Used when the main decremental algorithm cannot find alternative paths for certain edges.
 * 
 * Purpose:
 * - Generate short connecting paths between vertices that need connectivity
 * - Ensure essential graph connectivity is preserved during aggressive sparsification
 * - Provide fallback mechanism when no long alternative paths exist
 * 
 * Algorithm:
 * 1. Start from source vertex
 * 2. Take exactly 3 random steps (fixed short walk)
 * 3. Record complete 3-vertex path
 * 4. All intermediate edges will be added to sparse graph for connectivity
 * 
 * @param all_data  Random walk computation state
 * @param G         GPU dual graph structure
 * @param n_samples Number of vertices needing heuristic recovery
 */
__global__ void
random_walk_small(
			Random_walk_data_decremental * all_data,
			GPU_Dual_Graph * G,
			unsigned int n_samples
			){
	// === Thread Setup ===
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Initialize per-thread random number generator
	curandState local_state = all_data->global_state[tid];
	curand_init(tid, 10, 7, &local_state);

	// Get source vertex for heuristic recovery
	unsigned int sourceIndex = G->heuristic_sample_nodes_device[blockIdx.x];
	unsigned int currentVertex = sourceIndex;
	unsigned int previousVertex = 0;
	unsigned int path [3];  // Fixed 3-step path

	// === STEP 1: Start Path from Source ===
	path[0] = sourceIndex;
	unsigned int degree = G->dense_degree_list_device[currentVertex];
	float * offset = G->dense_beg_ptr_device[currentVertex];
	unsigned int next = curand(&local_state) % degree;
	unsigned int nextVertex = static_cast<unsigned int>(offset[next]);
	path[1] = nextVertex;  // First random step

	// === STEP 2: Take Second Random Step ===
	previousVertex = currentVertex;
	currentVertex = nextVertex;

	degree = G->dense_degree_list_device[currentVertex];
	next = curand(&local_state) % degree;
	nextVertex = static_cast<unsigned int>(offset[next]);
	
	// Avoid backtracking to previous vertex
	if (nextVertex == previousVertex){
		nextVertex = static_cast<unsigned int>(offset[degree - 1]);
	}
	path[2] = nextVertex;  // Second random step

	// === STEP 3: Store Complete 3-Step Path ===
	// All edges in this path will be added to sparse graph for connectivity
	for (int i = 0; i < 3; i++){
		all_data->path_selected_device[tid * 3 + i] = path[i];
	}
			
}




/**
 * Main Function: Decremental Graph Sparsification Program
 * 
 * Usage: ./decremental_update <graph_name>
 * 
 * Arguments:
 * - graph_name: Dataset name (looks for files in ../dataset/<name>/)
 * 
 * Input Files:
 * - updated_adj_sparse.mtx: Current sparse graph (target for sparsification)
 * - updated_dense.mtx: Updated dense graph with candidate edges removed (used for connectivity testing)
 * - del.mtx: Candidate edges to consider for removal
 * 
 * Output Files:
 * - added_edges.mtx: Recovery edges added to maintain connectivity
 * 
 * Algorithm Flow:
 * 1. Load both updated dense and sparse graphs plus candidate edges
 * 2. Process candidates in batches using dual graph approach
 * 3. Remove edges from both graphs, test connectivity on updated dense graph
 * 4. Add recovery paths when no alternative connectivity exists
 * 5. Output final recovery edges for analysis
 */
int main(int argc, char *argv[])
{	
	//cp adj_dense_mix.mtx adj_sparse.mtx del.mtx ./../../.; ../../random_walk_GPU; cp added_edges.mtx ~/Downloads/dyGRASS_UP

    // === Command Line Argument Parsing ===
    if(argc != 2){cout<<"Input: ./decremental_update <1. graph name>\n";exit(0);}
	const char* graph_name = argv[1];
	// Construct file paths using dataset directory structure
	string sparse_graph_file = "../dataset/" + string(graph_name) + "/updated_adj_sparse.mtx";  // Current sparse graph
	string dense_graph_file = "../dataset/" + string(graph_name) + "/updated_dense.mtx";    // Updated dense graph
	string del_file = "../dataset/" + string(graph_name) + "/del.mtx";                   // Candidate edges to remove
	string output_file = "../dataset/" + string(graph_name) + "/added_edges.mtx";         // Recovery edges

    // === Algorithm Parameters ===
    int n_steps = 100;                  // Maximum steps per random walk
    int n_walkers = 512;                // Number of parallel walkers per candidate edge (must be multiple of 32)
	int n_samples_splits = 10;           // Divide candidates into batches for memory management
	int n_heuristic_walkers = 16;        // Walkers for short heuristic recovery paths

    // === PHASE 1: Load and Prepare Data ===
    
    // Load both updated dense and sparse graphs
	cout << "Read graph and sparsifier" << endl;
	CSRGraph dense_graph(dense_graph_file.c_str(), true, 0, 1);   // Updated dense graph: file_name, is_reverse, skip_head, weightFlag
	CSRGraph sparse_graph(sparse_graph_file.c_str(), true, 0, 1); // Sparse graph: same format

	// Load candidate edges for removal (attached to dense_graph for batch processing)
	cout << "Read targets" << endl;
    dense_graph.read_targets(del_file.c_str(), -1, 0, n_samples_splits); // all_line, no_weight, divisor
	

	// === PHASE 2: Setup Dual Graph System ===
	
	// Create dual graph manager for decremental processing
	GPU_Dual_Graph ggraph(dense_graph, sparse_graph);
	size_t old_edge_num = ggraph.sparse_map->size();  // Track initial sparse graph size
	
	// Setup GPU memory for dual graph structure
	GPU_Dual_Graph * host_graph = &ggraph;
	GPU_Dual_Graph * device_graph;
	HRR(cudaMalloc(&device_graph, sizeof(GPU_Dual_Graph)));
	
	// === PHASE 3: Process First Batch of Candidate Edges ===
	
	// Remove first batch of edges and filter for existing edges in sparse graph
	bool is_empty = host_graph->removeAndUpdateDelEdges();
	// Copy updated dual graph data to GPU
	host_graph->copyDualGraphToDevice();
	cout<< "Graph copied to GPU" << endl;

	// === PHASE 4: Configure CUDA Kernel Parameters ===
	unsigned int n_samples = ggraph.decremental_sample_num;  // Candidate edges per batch
	unsigned int n_threadsPerBlock = n_walkers;              // Parallel walks per candidate (must be multiple of 32)
	unsigned int n_blocksPerGrid = n_samples;                // One block per candidate edge
	int n_threads = n_blocksPerGrid * n_threadsPerBlock;     // Total GPU threads

	// === PHASE 5: Setup Random Walk Computation State ===
	Random_walk_data_decremental * host_data = new Random_walk_data_decremental(n_steps, n_samples, n_threads);
	Random_walk_data_decremental * device_data;
	HRR(cudaMalloc(&device_data, sizeof(Random_walk_data_decremental)));

	// Copy data structures to GPU
	HRR(cudaMemcpy(device_data, host_data, sizeof(Random_walk_data_decremental), cudaMemcpyHostToDevice));
	HRR(cudaMemcpy(device_graph, host_graph, sizeof(GPU_Dual_Graph), cudaMemcpyHostToDevice));
	
	
	// === PHASE 6: Execute Decremental Sparsification Algorithm ===
	
	// Adjust block count to actual filtered edges (edges that existed in sparse graph)
	n_blocksPerGrid = host_graph->filtered_del_sample_count;
	auto start = std::chrono::high_resolution_clock::now();
	
	// === Process First Batch with Main Random Walks ===
	// Launch parallel random walks on updated dense graph to find alternative paths
	random_walk_decremental<<<n_blocksPerGrid, n_threadsPerBlock>>>(device_data, device_graph, n_samples, n_steps);
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for GPU completion
	
	// Transfer results back and update sparse graph with recovery paths
	host_data->get_result();
	host_graph->incrementalUpdateSparse(host_data->path_selected, host_data->path_selected_flag, host_data->final_conductance, n_steps);
	
	// === Heuristic Recovery for Essential Connectivity ===
	if(host_graph->heuristic_sample_num > 0){
		// Perform short random walks for edges that need heuristic recovery
		random_walk_small<<<host_graph->heuristic_sample_num, n_heuristic_walkers>>>(device_data, device_graph, host_graph->heuristic_sample_num);
		checkCudaErrors(cudaDeviceSynchronize());
		
		// Process heuristic results and add recovery edges
		host_data->get_result();
		host_graph->heuristicRecovery(host_data->path_selected, n_heuristic_walkers);
	}


	// === Process Remaining Batches ===
	int count = 0;
	while (! is_empty){

		// Advance to next batch of candidate edges
		is_empty = host_graph->removeAndUpdateDelEdges();
		n_samples = host_graph->filtered_del_sample_count;  // May be smaller for final batch
		
		// Launch random walks for current batch
		random_walk_decremental<<<n_samples, n_threadsPerBlock>>>(device_data, device_graph, n_samples, n_steps);
		checkCudaErrors(cudaDeviceSynchronize());
		
		// Process results and update sparse graph with recovery paths
		host_data->get_result();
		host_graph->incrementalUpdateSparse(host_data->path_selected, host_data->path_selected_flag, host_data->final_conductance, n_steps);

		// Heuristic recovery for current batch
		if(host_graph->heuristic_sample_num > 0){
			// Short random walks for essential connectivity
			random_walk_small<<<host_graph->heuristic_sample_num, n_heuristic_walkers>>>(device_data, device_graph, host_graph->heuristic_sample_num);
			checkCudaErrors(cudaDeviceSynchronize());
			
			// Add heuristic recovery edges
			host_data->get_result();
			host_graph->heuristicRecovery(host_data->path_selected, n_heuristic_walkers);
		}
		
		// Progress tracking and safety check
		count++;
		if(count > n_samples_splits){
			cout << "break by count, check code" << endl;
			break;
		}
	}
	// === PHASE 7: Output Results and Cleanup ===
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

	// Report sparsification effectiveness
	size_t new_edge_num = host_graph->sparse_map->size();
	cout << "Increased edges number: " << new_edge_num - old_edge_num << endl;
	cout << "Density: " << (float)(new_edge_num - host_graph->vertex_num)/host_graph->vertex_num * 100<< " %" << endl;
	
	// Save recovery edges added during decremental processing
	// Note: These are edges that were added back to maintain connectivity
	// host_graph->sparse_map_to_1_based_mtx("decremental_updated_sparse.mtx");
	host_graph->save_added_edges(output_file.c_str());
	
	// Clean up allocated memory
	delete host_data;
	HRR(cudaFree(device_data));
	HRR(cudaFree(device_graph));
	
	cout << "Decremental graph sparsification completed successfully!" << endl;
   return 0;
}
/**
 * dyGRASS Random Walk Incremental Graph Sparsification - Main CUDA Implementation
 * 
 * This program implements incremental graph sparsification using random walk sampling:
 * 
 * Algorithm Overview:
 * 1. Load original graph and extension edges (sample edges to test)
 * 2. For each sample edge (src, dest):
 *    - Launch parallel random walks from src trying to reach dest
 *    - If walks find connecting paths: edge is "sparsified" (not added)
 *    - If no paths found: edge must be added to maintain connectivity
 * 3. Process samples in batches to manage GPU memory
 * 4. Output final sparsified graph
 * 
 * Key Innovation: Only add edges when random walks fail to find alternative paths,
 * resulting in graphs that preserve connectivity with fewer edges.
 * 
 * CUDA Implementation:
 * - Each CUDA block processes one sample edge
 * - Multiple threads per block perform parallel random walks
 * - Reduction finds best path (highest conductance) among threads
 * - Batched processing handles large datasets efficiently
 */

#include <stdio.h>
#include <string.h>
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

// Maximum steps allowed per random walk (prevents infinite loops)
#define max_steps 100
using namespace std;

/**
 * CUDA Kernel: Parallel Random Walk Path Finding
 * 
 * This kernel is the core of the dyGRASS algorithm. Each CUDA block processes
 * one sample edge by launching multiple random walks in parallel.
 * 
 * Thread Organization:
 * - Each block handles one sample edge (src -> dest)
 * - Multiple threads per block perform independent random walks
 * - Threads use reduction to find the best path (highest conductance)
 * 
 * Random Walk Algorithm:
 * 1. Start at source vertex
 * 2. Randomly select neighbors (avoid returning to previous vertex)
 * 3. Accumulate resistance (1/weight) along path
 * 4. Stop when: target found, distortion exceeded, or max steps reached
 * 5. Calculate conductance = 1/total_resistance for successful paths
 * 
 * @param all_data  Random walk computation state and results
 * @param G         GPU graph structure with adjacency data
 * @param n_samples Number of sample edges in current batch
 * @param distortion Maximum allowed distortion (resistance * edge_weight)
 */
__global__ void
random_walk(
			Random_walk_data * all_data,
			gpu_graph * G,
			unsigned int n_samples, 
			float distortion
			){

	// === Thread and Sample Setup ===
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Initialize per-thread random number generator
	curandState local_state = all_data->global_state[tid];
	curand_init(tid, 10, 7, &local_state); 

	// Get sample edge endpoints from specialized layout
	unsigned int sourceIndex = G->sample_nodes_device[blockIdx.x];           // First half: sources
	unsigned int targetIndex = G->sample_nodes_device[blockIdx.x+n_samples]; // Second half: destinations
	float edge_weight = G->sample_weights_device[blockIdx.x];
	
	// === Random Walk State ===
	unsigned int currentVertex = sourceIndex;
	unsigned int previousVertex = 0;
	int targetFoundAt = -1;        // Step number where target was found (-1 if not found)
	int step_count = 1;            // Start counting from source (+1)
	float total_R = 0;             // Total resistance (sum of 1/weight) along path
	
	// === STEP 1: Select First Neighbor ===
	// Random walks begin by choosing a random neighbor from source
	unsigned int degree = G->degree_list_device[currentVertex];
	float * offset = G->beg_ptr_device[currentVertex];  // Points to neighbor data block
	unsigned int next = curand(&local_state) % degree;  // Random neighbor selection

	// Extract neighbor and weight from blocked memory layout
	unsigned int nextVertex = static_cast<unsigned int>(offset[next]);
	float weight = offset[next + degree];  // Weights stored offset by degree
	total_R += 1/weight;  // Accumulate resistance (1/weight)

	// === STEP 2: Check if Target Reached Immediately ===
	if (nextVertex != targetIndex){
		
		// === STEP 3: Continue Random Walk Until Termination ===
		previousVertex = currentVertex;
		currentVertex = nextVertex;
		step_count++;

		while(step_count < max_steps ){

			degree = G->degree_list_device[currentVertex];
			if (degree == 1){
				break; // Reached dead end (leaf node)
			}

			// === Neighbor Selection with Non-Backtracking ===
			offset = G->beg_ptr_device[currentVertex];
			// Select from degree-1 neighbors (exclude returning to previous)
			unsigned int next = curand(&local_state) % (degree - 1);
			weight_t inter = G->beg_ptr_device[currentVertex][next];
			nextVertex = static_cast<unsigned int>(inter);

			// If selected neighbor is previous vertex, choose last neighbor instead
			if (nextVertex == previousVertex){
				nextVertex = static_cast<unsigned int>(offset[degree - 1]);
				weight = offset[2*degree - 1];  // Weight block offset by 2*degree
			}
			else{
				weight = offset[next + degree];
			}

			total_R += 1/weight;

			// === Termination Conditions ===
			
			// Check distortion limit: total_resistance * edge_weight >= threshold
			if (total_R * edge_weight >= distortion){
				break; // Path too costly, abandon walk
			}

			// Check if target reached
			if (nextVertex == targetIndex){
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
	// This represents the best path found for this sample edge
	
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
			all_data->conductance_index_shared_mem[tid] = isSmall ? 
				all_data->conductance_index_shared_mem[tid + s] : 
				all_data->conductance_index_shared_mem[tid];
		}
		__syncthreads();
	}

	// === STEP 6: Winner Thread Records Final Result ===
	// Thread with best conductance writes result for this sample edge
	int max_conductance_thread = all_data->conductance_index_shared_mem[blockDim.x * blockIdx.x];
	if (threadIdx.x == max_conductance_thread){
		// Record whether path was found (step count or -1)
		all_data->path_selected_flag_device[blockIdx.x] = targetFoundAt;
		
		if (targetFoundAt > 0){ // Positive means target was found
			// Record the conductance of the best path
			all_data->final_conductance_device[blockIdx.x] = 1/total_R;
		}
	}
};





/**
 * Main Function: Incremental Graph Sparsification Program
 * 
 * Usage: ./incremental_update <graph_name> <distortion_threshold> <num_walkers>
 * 
 * Arguments:
 * - graph_name: Dataset name (looks for files in ../dataset/<name>/)
 * - distortion_threshold: Maximum allowed path distortion (float)
 * - num_walkers: Number of parallel random walks per sample (multiple of 32)
 * 
 * Input Files:
 * - adj_sparse.mtx: Original graph adjacency matrix
 * - ext.mtx: Extension edges to test for sparsification
 * 
 * Output Files:
 * - updated_adj_sparse.mtx: Final sparsified graph
 * 
 * Algorithm Flow:
 * 1. Load graph and sample edges
 * 2. Process samples in batches with parallel random walks
 * 3. Add edges only when no connecting paths are found
 * 4. Output final sparsified graph with connectivity preserved
 */
int main(int argc, char *argv[])
{
    // === Command Line Argument Parsing ===
    if(argc != 4){cout<<"Input: ./incremental_update <1.graph name> <2.distortion_threshold> <3.# of walkers>\n";exit(0);}
    
	const char* graph_name = argv[1];
	// Construct file paths using dataset directory structure
	string graph_file = "../dataset/" + string(graph_name) + "/adj_sparse.mtx";  // Original graph
	string target_file = "../dataset/" + string(graph_name) + "/ext.mtx";        // Sample edges
	string output_file = "../dataset/" + string(graph_name) + "/updated_adj_sparse.mtx"; // Result
	
    float distortion = atof(argv[2]);    // Distortion threshold for random walks
    int n_walkers = atoi(argv[3]);       // Number of parallel walkers (must be multiple of 32)

	
    // === PHASE 1: Load and Prepare Data ===
    
    // Load original graph with undirected edges (is_reverse=true), no header skip, weighted
    CSRGraph graph = read_graph(graph_file.c_str(), true, 0, 1);
    
    // Load sample edges to test for sparsification
    Targets targets = read_targets(target_file.c_str(), -1, 1);  // Process all edges, weighted
	cout << "after read_targets" << endl;
	
    // Attach sample data to graph structure
    graph.sample_count = targets.target_count;
    graph.sample_nodes = targets.nodes;     // Special layout: [src1,src2,...,dest1,dest2,...]
	graph.sample_weights = targets.weights;
    graph.to_pointer();  // Convert degree array to pointers (if needed)

	// === PHASE 2: Setup GPU Data Structures ===
	
	// Create GPU-optimized graph structure
	gpu_graph ggraph(graph, distortion);
	gpu_graph * host_graph = &ggraph;
	gpu_graph * device_graph;
	HRR(cudaMalloc(&device_graph, sizeof(gpu_graph)));
	
	// === PHASE 3: Configure CUDA Kernel Parameters ===
	unsigned int n_samples = ggraph.sample_count;   // Edges per batch (~total/10)
	unsigned int n_threadsPerBlock = n_walkers;     // Parallel walks per sample
	unsigned int n_blocksPerGrid = n_samples;       // One block per sample edge
	int n_threads = n_blocksPerGrid * n_threadsPerBlock; // Total GPU threads

	// === PHASE 4: Setup Random Walk Computation State ===
	Random_walk_data * host_data = new Random_walk_data(n_samples, n_threads);
	Random_walk_data * device_data;

	// Copy data structures to GPU
	HRR(cudaMalloc(&device_data, sizeof(Random_walk_data)));
	HRR(cudaMemcpy(device_data, host_data, sizeof(Random_walk_data), cudaMemcpyHostToDevice));
	ggraph.copyToDevice();  // Transfer graph data to GPU
	HRR(cudaMemcpy(device_graph, host_graph, sizeof(gpu_graph), cudaMemcpyHostToDevice));
	cout<< "Graph copied to GPU" << endl;

	// === PHASE 5: Execute Incremental Sparsification Algorithm ===
	
	auto start = std::chrono::high_resolution_clock::now();
	
	// === Process First Batch ===
	// Launch parallel random walks for first batch of sample edges
	random_walk<<<n_blocksPerGrid, n_threadsPerBlock>>>(device_data, device_graph, n_samples, distortion);
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for GPU completion
	
	// Transfer results back to host and update graph
	host_data->get_result();
	host_graph->updateGraphFromResult(
		host_data->path_selected_flag,  // -1 if no path, step count if found
		host_data->final_conductance    // Conductance of best paths
	);

	// === Process Remaining Batches ===
	bool sample_nodes_exist = true;
	int count = 0;
	while (sample_nodes_exist){

		// Advance to next batch of sample edges
		sample_nodes_exist = host_graph->updateSampleEdges();
		n_samples = host_graph->sample_count;  // May be smaller for final batch
		
		// Launch random walks for current batch
		random_walk<<<n_samples, n_threadsPerBlock>>>(device_data, device_graph, n_samples, distortion);
		
		// Process results and update graph
		host_data->get_result();
		host_graph->updateGraphFromResult(
			host_data->path_selected_flag, 
			host_data->final_conductance
		);
		
		// Progress tracking
		float processed = (float)(host_graph->sample_count + host_graph->sample_ptr/2)/host_graph->sample_total_count * 100;
		float extra_memory_usage = (float)(host_graph->extra_offset/4)/host_graph->edge_count * 100;

		count++;
		if (count > 10){
			// Safety check to prevent infinite loops during debugging
			cout << "Reached the maximum number of iterations" << endl;
			break;
		}
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

	// === PHASE 6: Output Results and Cleanup ===
	
	// Report sparsification effectiveness
	cout << "No path %: " << (float)host_graph->no_path_count/host_graph->sample_total_count * 100 << endl;
	cout << "Edges added: " << host_graph->no_path_count << " out of " << host_graph->sample_total_count << " samples" << endl;
	
	// Save final sparsified graph (original + necessary edges)
	host_graph->saveCurrentMtx(output_file.c_str());
	
	// Clean up allocated memory
	delete host_data;
	HRR(cudaFree(device_data));
	HRR(cudaFree(device_graph));
	
	cout << "Incremental graph sparsification completed successfully!" << endl;
	return 0;
}
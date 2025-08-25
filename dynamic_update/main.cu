/**
 * @file main.cu
 * @brief Main driver program for dynamic graph sparsification using GPU-accelerated random walks
 * 
 * This program implements the dyGRASS (Dynamic Graph Spectral Sparsification) algorithm
 * for maintaining sparse approximations of dynamic graphs through incremental and decremental
 * edge updates using neighborhood-based random walks (NBRW).
 * 
 * Program Flow:
 * 1. Load dense and sparse graph representations from MTX files
 * 2. Initialize GPU data structures and memory management
 * 3. Process edge update batches sequentially:
 *    - Load next batch from stream files  
 *    - Execute appropriate CUDA kernel (incremental/decremental)
 *    - Update sparsifier based on random walk results
 *    - Optionally check graph properties and condition numbers
 * 4. Generate timestamped output with final results
 * 
 * Command Line Usage:
 *   ./dynamic_update <graph_name> <distortion_threshold> [inc_steps] [dec_steps] [inc_walkers] [dec_walkers]
 * 
 * Input Files Expected:
 *   - ./dataset/<graph_name>/new_adj_dense.mtx (dense graph)
 *   - ./dataset/<graph_name>/new_adj_sparse.mtx (initial sparse graph)  
 *   - ./dataset/<graph_name>/stream_edges/ (directory with batch files)
 * 
 * Key Features:
 *   - GPU-accelerated random walk sampling with 512 walkers per edge
 *   - Interactive batch processing with user prompts
 *   - Real-time condition number analysis via Julia integration
 *   - Professional output formatting with timing information
 *   - Memory-efficient batch processing with overflow handling
 * 
 * @author Yihang Yuan
 * @date 2025
 */

#include <stdio.h>
#include <string.h>
// #include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
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
// #define max_steps 100 // already defined in functions.h
using namespace std;

/**
 * @brief Main function - Dynamic Graph Sparsification Pipeline
 * 
 * Orchestrates the complete dynamic sparsification workflow:
 * 1. Parameter validation and configuration
 * 2. Graph loading and GPU initialization
 * 3. Batch processing loop with kernel execution
 * 4. Results collection and output generation
 * 
 * @param argc Argument count (3 or 7 expected)
 * @param argv Command line arguments:
 *        argv[1]: Graph name (dataset subdirectory)
 *        argv[2]: Distortion threshold for random walks
 *        argv[3-6]: Optional - custom walker counts and step limits
 * @return 0 on successful completion
 */
int main(int argc, char *argv[]){

    if(argc != 7 && argc != 3){cout<<"Input: .dynamic_update" 
        << "<1.graph name>"
        << "<2.distortion_threshold>" 
        << "<3.incremental_steps>"
        << "<4.decremental_steps>"
        << "<5.incremental_number_of_walkers>"
        << "<6.decremental_number_of_walkers>"
        << endl;
        exit(0);
    }

    string timestamp = getCurrentTimestamp();
    const char* graph_name = argv[1];
    string sparse_graph_name = "./dataset/" + string(graph_name) + "/new_adj_sparse.mtx";
    string dense_graph_name = "./dataset/" + string(graph_name) + "/new_adj_dense.mtx";
    string stream_edge_folder = "./dataset/" + string(graph_name) + "/stream_edges/";
    string output_folder = "./output/" + string(graph_name) + "/" + timestamp ;

    float distortion = atof(argv[2]);
    int n_steps_incremental, n_steps_decremental, n_walkers_incremental, n_walkers_decremental;

    if (argc == 3){
        n_steps_incremental = max_steps;
        n_steps_decremental = max_steps;
        n_walkers_incremental = 512;
        n_walkers_decremental = 512;
    }else if (argc == 7) {
        n_steps_incremental =  atoi(argv[3]);
        n_steps_decremental =  atoi(argv[4]);
        n_walkers_incremental = atoi(argv[5]);
        n_walkers_decremental = atoi(argv[6]);
    }

    cout << "Graph Name: " << graph_name << endl;
    cout << "Distortion Threshold: " << distortion << endl;
    cout << "Incremental Steps: " << n_steps_incremental << endl;
    cout << "Decremental Steps: " << n_steps_decremental << endl;
    cout << "Incremental Number of Walkers: " << n_walkers_incremental << endl;
    cout << "Decremental Number of Walkers: " << n_walkers_decremental << endl;

    cout << "Read graph and sparsifier" << endl;
    cout << "Dense graph reading..." << endl;
    CSRGraph dense_graph(dense_graph_name.c_str());
    cout << "Sparse graph reading..." << endl;
    CSRGraph sparse_graph(sparse_graph_name.c_str());
    assert(dense_graph.base == sparse_graph.base);
    assert(dense_graph.v_max == sparse_graph.v_max);
    assert(dense_graph.v_min == sparse_graph.v_min);
    cout << "Dual graph class construction..." << endl;
    GPU_Dual_Graph gpu_graph(&dense_graph, &sparse_graph);
    GPU_Dual_Graph * host_graph_ptr = &gpu_graph;
    GPU_Dual_Graph * device_graph_ptr;
    HRR(cudaMalloc(&device_graph_ptr, sizeof(GPU_Dual_Graph)));
    HRR(cudaMemcpy(device_graph_ptr, host_graph_ptr, sizeof(GPU_Dual_Graph), cudaMemcpyHostToDevice));

    int stream_edge_batch_max_capacity = dense_graph.vert_count * 0.05; //maximum 5% density for each batch
    if (stream_edge_batch_max_capacity < 10000) stream_edge_batch_max_capacity = 10000;
    EdgeStream edge_stream(stream_edge_folder.c_str(), dense_graph.base);

    GPU_Stream_Edges gpu_stream_edges(stream_edge_batch_max_capacity, max_steps);
    GPU_Stream_Edges * host_stream_edges_ptr = &gpu_stream_edges;
    GPU_Stream_Edges * device_stream_edges_ptr;
    HRR(cudaMalloc(&device_stream_edges_ptr, sizeof(GPU_Stream_Edges)));
    HRR(cudaMemcpy(device_stream_edges_ptr, host_stream_edges_ptr, sizeof(GPU_Stream_Edges), cudaMemcpyHostToDevice));

    int batch_counter = 0;
    double total_kernel_time = 0.0;

    char check_properties;
    cout << "Do you want to check initial graph properties? (y/n): ";
    cin >> check_properties;
    // check_properties = 'y'; // disable property check by default
    if (check_properties == 'y' || check_properties == 'Y') {
        host_graph_ptr->check_current_properties();
    }

    while (edge_stream.loadNextBatch()){
        batch_counter++;
        
        // Batch header is now printed by loadNextBatch()
        
        host_graph_ptr->preprocessStreamEdges(edge_stream);
        host_stream_edges_ptr->loadEdgeFromStream(edge_stream);
        
        cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << endl;
        cout << "│ Processing Batch " << std::setw(2) << batch_counter 
             << " │ Operation: " << std::setw(11) << (host_stream_edges_ptr->current_op == INCREMENTAL ? "INCREMENTAL" : "DECREMENTAL") 
             << " │ Edges: " << std::setw(4) << host_stream_edges_ptr->batch_size << " │" << endl;
        cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << endl;

        // Prompt user to check current graph properties
        

        do {
            host_stream_edges_ptr->loadEdgesToDevice();
            unsigned int n_blockPerGrid = host_stream_edges_ptr->load_size;

            if (host_stream_edges_ptr->current_op == INCREMENTAL){
                cout << ">>> Incremental kernel is running..." << endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                NBRW_incremental<<<n_blockPerGrid,  n_walkers_incremental>>>(device_graph_ptr, device_stream_edges_ptr, distortion, n_steps_incremental);
                cudaDeviceSynchronize();
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                double kernel_time = duration.count() / 1000.0;
                total_kernel_time += kernel_time;
                cout << ">>> Incremental kernel completed in " << std::fixed << std::setprecision(3) << kernel_time << " seconds" << endl;
            }else if (host_stream_edges_ptr->current_op == DECREMENTAL){
                cout << ">>> Decremental kernel is running..." << endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                NBRW_decremental<<<n_blockPerGrid,  n_walkers_decremental>>>(device_graph_ptr, device_stream_edges_ptr, n_steps_decremental);
                cudaDeviceSynchronize();
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                double kernel_time = duration.count() / 1000.0;
                total_kernel_time += kernel_time;
                cout << ">>> Decremental kernel completed in " << std::fixed << std::setprecision(3) << kernel_time << " seconds" << endl;
            }
            
            // Force flush kernel printf output
            fflush(stdout);

            cout << "Processing kernel results and updating sparsifier..." << endl;
            host_graph_ptr->updateSparsiferFromResult(gpu_stream_edges);
            
        }while (host_stream_edges_ptr->overflow_flag == true);
        
        cout << "Batch " << batch_counter << " completed successfully!" << endl;

        cout << "Do you want to check current graph properties? (y/n): ";
        cin >> check_properties;
        // check_properties = 'y'; // disable property check by default
        if (check_properties == 'y' || check_properties == 'Y') {
            host_graph_ptr->check_current_properties();
        }

        cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
        cout << endl;
        
    }

    cout << "\n╔══════════════════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║                                PROCESSING COMPLETE                               ║" << endl;
    cout << "║                                                                                  ║" << endl;
    // Create properly aligned total kernel runtime line (total width = 85 chars including both borders)
    ostringstream runtime_stream;
    runtime_stream << std::fixed << std::setprecision(3) << total_kernel_time;
    string runtime_line = "║ Total Kernel Runtime: " + runtime_stream.str() + " seconds";
    cout << runtime_line << string(85 - runtime_line.length(), ' ') << "║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════════════════════════╝" << endl;
    
    // Prompt user to check final graph properties
    cout << "Do you want to check final graph properties? (y/n): ";
    cin >> check_properties;
    
    if (check_properties == 'y' || check_properties == 'Y') {
        cout << "\n┌─────────────────────────────────────────────────────────────────────────────────┐" << endl;
        cout << "│                              FINAL GRAPH ANALYSIS                               │" << endl;
        cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << endl;
        host_graph_ptr->check_current_properties();
    }

    cout << "Saving results to: " << output_folder << endl;
    host_graph_ptr->save_result(output_folder);
    
    cout << "Dynamic graph sparsification completed successfully!" << endl;
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
    cout << endl;


    return 0;
}
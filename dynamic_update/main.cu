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



// Maximum steps allowed per random walk (prevents infinite loops)
// #define max_steps 100 // already defined in functions.h
using namespace std;

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

    while (edge_stream.loadNextBatch()){

        // Prompt user to check current graph properties
        char check_properties;
        cout << "\nDo you want to check current graph properties? (y/n): ";
        cin >> check_properties;
        // check_properties = 'y'; // disable property check by default
        if (check_properties == 'y' || check_properties == 'Y') {
            host_graph_ptr->check_current_properties();
        }

        host_graph_ptr->preprocessStreamEdges(edge_stream);
        host_stream_edges_ptr->loadEdgeFromStream(edge_stream);

        do {
            host_stream_edges_ptr->loadEdgesToDevice();
            unsigned int n_blockPerGrid = host_stream_edges_ptr->load_size;

            if (host_stream_edges_ptr->current_op == INCREMENTAL){
                cout << "Incremental kernel" << endl;
                NBRW_incremental<<<n_blockPerGrid,  n_walkers_incremental>>>(device_graph_ptr, device_stream_edges_ptr, distortion, n_steps_incremental);
            }else if (host_stream_edges_ptr->current_op == DECREMENTAL){
                cout << "Decremental kernel" << endl;
                NBRW_decremental<<<n_blockPerGrid,  n_walkers_decremental>>>(device_graph_ptr, device_stream_edges_ptr, n_steps_decremental);
            }
            
            // Force flush kernel printf output
            cudaDeviceSynchronize();
            fflush(stdout);

            cout << "Download result" << endl;
            host_graph_ptr->updateSparsiferFromResult(gpu_stream_edges);
            
        }while (host_stream_edges_ptr->overflow_flag == true);
        
    }

    

    host_graph_ptr->save_result(output_folder);



    return 0;
}
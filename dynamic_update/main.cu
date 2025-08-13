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

__global__ void
NBRW_decremental(
    GPU_Dual_Graph * G,
    GPU_Stream_Edges * stream_edges,
    int batch_size,
    int n_steps
){}

__global__ void
NBRW_incremental(
    GPU_Dual_Graph * G,
    GPU_Stream_Edges * stream_edges,
    int batch_size,
    float distortion
){}


__global__ void
NBRW_heuristic_decremental(
    GPU_Dual_Graph * G,
    GPU_Stream_Edges * stream_edges,
    int batch_size
){}

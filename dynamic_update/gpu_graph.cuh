
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
#include <unordered_map>  // for O(1) edge mapping
#include <unordered_set>  // for tracking added edges

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




#endif
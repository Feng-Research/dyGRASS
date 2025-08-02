/**
 * @file functions.h
 * @brief Core data structures and function declarations for graph processing
 * 
 * This file defines the main data structures used for representing graphs
 * in Compressed Sparse Row (CSR) format and managing target edges for
 * incremental random walk-based graph sparsification.
 */

#ifndef FUNCTION_H
#define FUNCTION_H

#include<iostream>
#include<tuple>
#include<fstream>
#include <sys/stat.h> // for stat
#include <sys/mman.h> // for mmap
#include <fcntl.h> // for open
#include <assert.h> // for assert
#include <vector>
#include <unistd.h> // for close

using namespace std;

// Large number representing infinity for graph algorithms
#define INFTY int(1<<30)

// Type definitions for better code readability and consistency
typedef unsigned int vertex_t;  // Vertex identifier type
typedef unsigned int index_t;   // Array index type
typedef float weight_t;         // Edge weight type

/**
 * @class CSRGraph
 * @brief Compressed Sparse Row (CSR) representation of a graph
 * 
 * This class stores a graph in CSR format, which is efficient for sparse graphs.
 * It provides both the basic CSR data (adj, begin, degree) and additional
 * metadata needed for random walk-based sparsification.
 */
class CSRGraph {
    public:
    // Core CSR representation
    vector<vertex_t> adj;        // Flattened adjacency list: neighbors of all vertices
    vector<weight_t> weight;     // Edge weights corresponding to adj entries
    vector<index_t> begin;       // Starting index in adj array for each vertex
    vector<index_t> degree;      // Number of neighbors for each vertex
    
    // Additional metadata for sparsification
    vector<vertex_t> from;       // Original edge index from MTX file
    vector<vertex_t> reverse;    // Reverse edge lookup for undirected graphs
    vector<tuple<vertex_t, vertex_t, weight_t>> mtx; // Original edge list format
    
    // Graph statistics
    size_t edge_count;           // Total number of directed edges
    size_t vert_count;           // Total number of vertices
    size_t sample_count;         // Number of target edges for sampling
    vertex_t v_max;              // Maximum vertex ID in the graph
    vertex_t v_min;              // Minimum vertex ID in the graph

    // Raw pointers for efficient GPU data transfer
    vertex_t* sample_nodes;      // Array of source/target pairs for random walks
    weight_t* sample_weights;    // Weights of edges to be sampled
    vertex_t* adj_list;          // Raw pointer to adj vector data
    weight_t* weight_list;       // Raw pointer to weight vector data
    index_t* beg_pos;            // Raw pointer to begin vector data
    vertex_t* degree_list;       // Raw pointer to degree vector data

    CSRGraph() : edge_count(0), vert_count(0), sample_count(0), v_max(0), v_min(INFTY), sample_nodes(nullptr) {}

    ~CSRGraph() {
        cout <<"CSRGraph deleted" << endl;
    }

    /**
     * @brief Convert vector data to raw pointers for GPU transfer
     * 
     * This method extracts raw pointers from STL vectors to facilitate
     * efficient data transfer to GPU memory.
     */
    void to_pointer(){
        adj_list = adj.data();
        weight_list = weight.data();
        beg_pos = begin.data();
        degree_list = degree.data();
    }


};

/**
 * @struct Targets
 * @brief Container for edge targets used in incremental sparsification
 * 
 * This structure holds the list of edges to be processed during incremental
 * graph updates. The nodes array is organized as [sources..., targets...]
 * where the first half contains source vertices and the second half contains
 * corresponding target vertices.
 */
struct Targets {
    vertex_t* nodes;     // Array layout: [src1,src2,...,tgt1,tgt2,...]
    weight_t* weights;   // Corresponding edge weights
    size_t target_count; // Number of edges to process

    ~Targets() {
        delete[] nodes;
        delete[] weights;
        cout <<"Targets deleted" << endl;
    }
};

/**
 * @brief Read a graph from MTX format file into CSR representation
 * @param filename Path to the MTX format graph file
 * @param is_reverse Whether to create reverse edges (for undirected graphs)
 * @param skip_head Number of header lines to skip
 * @param weightFlag Whether the file contains edge weights (0=no, 1=yes)
 * @return CSRGraph object containing the loaded graph
 */
CSRGraph read_graph(const char* filename, bool is_reverse, long skip_head, int weightFlag);

/**
 * @brief Read target edges for incremental sparsification
 * @param filename Path to the file containing new edges to add
 * @param process_line Maximum number of lines to process (-1 for all)
 * @param weightFlag Whether the file contains edge weights
 * @return Targets structure containing source-target pairs and weights
 */
Targets read_targets(const char* filename, int process_line = -1, int weightFlag = 1);

#endif
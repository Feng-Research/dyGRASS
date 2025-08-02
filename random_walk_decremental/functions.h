
/**
 * Header file for dyGRASS Random Walk Decremental Graph Sparsification
 * 
 * This header defines core data structures and functions for decremental
 * graph sparsification, which removes edges from a graph while preserving
 * connectivity through random walk sampling.
 * 
 * Key Differences from Incremental Version:
 * - Starts with fully connected graph and removes edges
 * - Uses edge mapping for efficient edge deletion
 * - Processes batches of edges to remove (rather than add)
 * - Maintains connectivity by checking if remaining paths exist
 * 
 * Data Structures:
 * - CSRGraph: Enhanced with edge deletion capabilities
 * - Targets: Edges to consider for removal
 * - Edge mapping for O(1) edge lookup and deletion
 */

#ifndef FUNCTION_H
#define FUNCTION_H

#include<iostream>
#include<tuple>
#include<fstream>
#include <sys/stat.h> // for stat - file size information
#include <sys/mman.h> // for mmap - memory mapping files
#include <fcntl.h> // for open - file descriptor operations
#include <assert.h> // for assert - debug assertions
#include <vector>
#include <unistd.h> // for close - closing file descriptors
#include <unordered_map> // for efficient edge mapping

using namespace std;

#define INFTY int(1<<30)  // Large number representing infinity

// Type aliases for clarity and consistency
typedef unsigned int vertex_t;  // Vertex identifier type
typedef unsigned int index_t;   // Array index type
typedef float weight_t;         // Edge weight type

/**
 * Target Edges Structure for Decremental Processing
 * 
 * Contains edges that are candidates for removal during decremental
 * sparsification. These edges will be tested with random walks to
 * determine if they can be safely removed while preserving connectivity.
 * 
 * Memory Layout:
 * - First half of nodes array: source vertices
 * - Second half of nodes array: destination vertices
 * - Weights array: corresponding edge weights
 */
struct Targets {
    vertex_t* nodes;     // Edge endpoints [src1,src2,...,srcN,dest1,dest2,...,destN]
    weight_t* weights;   // Edge weights [w1, w2, ..., wN]
    size_t target_count; // Number of target edges to process

    /**
     * Destructor: Clean up allocated memory
     */
    ~Targets() {
        delete[] nodes;
        delete[] weights;
        cout <<"Targets deleted" << endl;
    }
};

/**
 * Enhanced CSR Graph for Decremental Sparsification
 * 
 * Extends standard Compressed Sparse Row format with:
 * - Edge mapping for O(1) edge deletion
 * - Reverse edge tracking for bidirectional updates
 * - Sample edge management for batch processing
 * - Edge removal capabilities while maintaining CSR structure
 * 
 * Key Enhancement: edge_map provides fast lookup from (src,dest) -> array positions
 * This enables efficient edge deletion without full array reconstruction.
 */
class CSRGraph {
    public:
        // === Core CSR Graph Data ===
        vector<vertex_t> adj;        // Adjacency list: all neighbors concatenated
        vector<weight_t> weight;     // Edge weights corresponding to adj entries
        vector<index_t> begin;       // Starting index in adj[] for each vertex
        vector<index_t> degree;      // Number of neighbors for each vertex
        vector<vertex_t> from;       // Original edge index mapping
        vector<vertex_t> reverse;    // Reverse edge position mapping
        vector<tuple<vertex_t, vertex_t, weight_t>> mtx; // Original edge tuples
        
        // === Edge Deletion Support ===
        unordered_map<long, pair<index_t,index_t>> edge_map; // (src,dest) -> (pos1,pos2) for O(1) deletion
        
        // === Graph Statistics ===
        size_t line_count;           // Number of original edges (undirected pairs)
        size_t edge_count;           // Total directed edges (line_count * 2)
        size_t vert_count;           // Number of vertices
        vertex_t v_max;              // Maximum vertex ID
        vertex_t v_min;              // Minimum vertex ID
        int multiplier;              // Internal scaling factor
        
        // === Raw Pointer Access (for GPU transfer) ===
        vertex_t* adj_list;          // Pointer to adj vector data
        weight_t* weight_list;       // Pointer to weight vector data
        index_t* beg_pos;            // Pointer to begin vector data
        vertex_t* degree_list;       // Pointer to degree vector data
        
        // === Sample Edge Data for Batch Processing ===
        size_t sample_count;         // Number of sample edges per batch
        vertex_t* sample_nodes;      // Sample edge endpoints for random walks
        weight_t* sample_weights;    // Weights of sample edges
        int divisor;                 // Batch division factor

        
    // === Constructors and Destructor ===
    
    /**
     * Default constructor: Initialize empty graph
     */
    CSRGraph() : edge_count(0), vert_count(0), sample_count(0), v_max(0), v_min(INFTY), sample_nodes(nullptr) {}

    /**
     * File constructor: Load graph from file
     * @param filename Path to graph file
     * @param is_reverse Create undirected graph (add reverse edges)
     * @param skip_head Number of header lines to skip
     * @param weightFlag 1 if file contains weights, 0 for unweighted
     */
    CSRGraph(const char* filename, bool is_reverse, long skip_head, int weightFlag);

    /**
     * Destructor: Clean up resources
     */
    ~CSRGraph() {
        cout <<"CSRGraph deleted" << endl;
    }

    // === Utility Methods ===
    
    /**
     * Convert vector data to raw pointers for GPU transfer
     */
    void to_pointer(){
        adj_list = adj.data();
        weight_list = weight.data();
        beg_pos = begin.data();
        degree_list = degree.data();
    }

    /**
     * Count number of digits in integer (utility function)
     */
    int findDigitsNum(int n){
        int count = 0;
        while (n != 0){
            n = n/10;
            count++;
        }
        return count;
    }
    
    // === Edge Management Methods ===
    
    /**
     * Load target edges for decremental processing
     * @param filename Path to target edges file
     * @param process_line Number of edges to process (-1 for all)
     * @param weightFlag 1 if file contains weights
     * @param divisor Batch division factor
     */
    void read_targets(const char* filename, int process_line, int weightFlag, int divisor);

    /**
     * Remove edges from graph based on target edge results
     * Core decremental operation: removes edges when random walks
     * find alternative paths, preserving only necessary edges.
     */
    void remove_edge_from_targets();


};


/**
 * Note: Main graph reading functionality is now integrated into CSRGraph constructor
 * and class methods. These commented declarations show the evolution from standalone
 * functions to object-oriented design.
 */

#endif
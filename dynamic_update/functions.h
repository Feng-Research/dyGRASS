/**
 * @file functions.h
 * @brief Unified data structures and function declarations for dynamic graph processing
 * 
 * This file defines the main data structures used for representing graphs
 * in Compressed Sparse Row (CSR) format and managing both incremental and
 * decremental random walk-based graph sparsification in a unified framework.
 * 
 * Combines features from both incremental and decremental implementations:
 * - CSR graph representation with efficient edge addition/removal
 * - Edge mapping for O(1) operations
 * - Batch processing for both insertion and deletion
 * - GPU memory management support
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
#include <unordered_map> // for efficient edge mapping
#include <string>

using namespace std;

// Large number representing infinity for graph algorithms
#define INFTY int(1<<30)

// Type definitions for better code readability and consistency
typedef unsigned int vertex_t;  // Vertex identifier type
typedef unsigned int index_t;   // Array index type
typedef float weight_t;         // Edge weight type

/**
 * @enum OperationType
 * @brief Type of graph operation to perform
 */
enum OperationType {
    INCREMENTAL,    // Add edges to graph
    DECREMENTAL     // Remove edges from graph
};



struct GraphInfo {
      long skip_lines;
      bool has_matrixmarket_header;
      bool has_dimensions_line;
      bool is_weighted;
      bool is_laplacian;
      bool is_triangle;
      int base;
      vertex_t v_max, v_min;
  };

class CSRGraph {
    public:
    // === Core CSR representation ===
    vector<vertex_t> adj;        // Flattened adjacency list: neighbors of all vertices
    vector<weight_t> weight;     // Edge weights corresponding to adj entries
    vector<index_t> begin;       // Starting index in adj array for each vertex
    vector<index_t> degree;      // Number of neighbors for each vertex
    
    // === Additional metadata for sparsification ===
    vector<vertex_t> from;       // Original edge index from MTX file
    vector<vertex_t> reverse;    // Reverse edge lookup for undirected graphs
    vector<tuple<vertex_t, vertex_t, weight_t>> mtx; // Original edge list format
    
    // === Edge mapping for efficient operations ===
    unordered_map<long, pair<index_t,index_t>> edge_map; // (src,dest) -> (pos1,pos2) for O(1) deletion
    
    // === Graph statistics ===
    size_t line_count;           // Number of original edges (undirected pairs)
    size_t edge_count;           // Total number of directed edges
    size_t vert_count;           // Total number of vertices
    vertex_t v_max;              // Maximum vertex ID in the graph
    vertex_t v_min;              // Minimum vertex ID in the graph
    long multiplier;             // Hash key multiplier for edge mapping
    
    // === Raw pointers for efficient GPU data transfer ===
    vertex_t* adj_list;          // Raw pointer to adj vector data
    weight_t* weight_list;       // Raw pointer to weight vector data
    index_t* beg_pos;            // Raw pointer to begin vector data
    vertex_t* degree_list;       // Raw pointer to degree vector data

    

    // === Constructors and Destructor ===
    CSRGraph() : line_count(0), edge_count(0), vert_count(0), v_max(0), v_min(INFTY), 
                 multiplier(1000000) {}

    /**
     * File constructor: Load graph from file
     * @param filename Path to graph file
     */
    CSRGraph(const char* filename);

    /**
     * Destructor: Clean up resources
     */
    ~CSRGraph() {
        cout <<"CSRGraph deleted" << endl;
    }

    // === Utility Methods ===
    
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

};


class EdgeStream {
    public:
    int base;

    string operations_folder; 
    vector<string> batch_names;
    size_t batch_index;                // Current position in sequence
    size_t batch_size;
    string batch_filename;
    vector<tuple<vertex_t, vertex_t, weight_t>> batch_edges; // Reference to current batch edges

    OperationType current_op;
    size_t total_edges_processed;             // Running count of processed edges
    vector<size_t> batch_sizes;               // Edge count per batch for analysis
    bool manual_selection;
    
    
    EdgeStream(const string& folder, int base);
    

    ~EdgeStream() {
        cout << "EdgeStream deleted. Processed " << total_edges_processed << " total edges." << endl;
    }
    

    bool loadNextBatch();

    bool autoDetectWeightExist(const string& filename);

    void loadBatchFromFile(const string& filename, OperationType op);
    

};



GraphInfo auto_detect_graph_info(const char* filename);

#endif
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


struct Targets {
    vertex_t* nodes;     // Array layout: [src1,src2,...,tgt1,tgt2,...]
    weight_t* weights;   // Corresponding edge weights
    size_t target_count; // Number of edges to process

    /**
     * Constructor: Initialize empty targets
     */
    Targets() : nodes(nullptr), weights(nullptr), target_count(0) {}

    /**
     * Destructor: Clean up allocated memory
     */
    ~Targets() {
        delete[] nodes;
        delete[] weights;
        cout <<"Targets deleted" << endl;
    }
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

/**
 * @struct BatchOperation
 * @brief Represents a single batch operation in the dynamic edge flow
 * 
 * Each batch operation contains a set of edges to be processed together
 * for either incremental (addition) or decremental (removal) sparsification.
 */
struct BatchOperation {
    int index;                    // Sequential batch index (0, 1, 2, ...)
    OperationType operation_type; // INCREMENTAL or DECREMENTAL
    string filename;              // Path to batch file
    size_t edge_count;           // Number of edges in this batch
    
    // Loaded edge data (organized for GPU processing)
    vertex_t* edges;             // Edge endpoints [src1,src2,...,dest1,dest2,...]
    weight_t* weights;           // Corresponding edge weights
    
    BatchOperation() : index(-1), operation_type(INCREMENTAL), edge_count(0), 
                      edges(nullptr), weights(nullptr) {}
    
    BatchOperation(int idx, OperationType op_type, const string& file) 
        : index(idx), operation_type(op_type), filename(file), edge_count(0),
          edges(nullptr), weights(nullptr) {}
    
    ~BatchOperation() {
        delete[] edges;
        delete[] weights;
    }
};

/**
 * @class EdgeStream
 * @brief Manages dynamic edge flow processing for both incremental and decremental operations
 * 
 * This class handles the file-based batch processing system for dynamic graph updates:
 * - Discovers and organizes batch operation files
 * - Loads edges from batch files in the correct format
 * - Manages the sequence of operations for reproducible experiments
 * - Provides interactive control over execution flow
 * 
 * Key Features:
 * - File naming convention: XX_inc_batch.mtx, XX_dec_batch.mtx
 * - Sequential processing with user control
 * - Memory efficient batch loading
 * - GPU-ready data organization
 */
class EdgeStream {
    public:
    int base;

    string operations_folder; 
    size_t batch_number;
    size_t current_batch_index;                // Current position in sequence
    size_t current_batch_size;
    BatchOperation * current_batch_edges; // Pointer to current batch edges
    OperationType current_op;

    
    // === Statistics ===
    size_t total_edges_processed;             // Running count of processed edges
    vector<size_t> batch_sizes;               // Edge count per batch for analysis
    
    // === Constructors and Destructor ===
    
    /**
     * Default constructor
     */
    EdgeStream() : current_batch_index(0), 
                   total_edges_processed(0) {}
    
    /**
     * Constructor with folder path
     * @param folder Path to directory containing batch files
     */
    EdgeStream(const string& folder, int base);
    
    /**
     * Destructor: Clean up resources
     */
    ~EdgeStream() {
        cout << "EdgeStream deleted. Processed " << total_edges_processed << " total edges." << endl;
    }
    
    // === Core Functionality ===

    
    /**
     * Load next batch operation from sequence
     * @return Pointer to loaded batch operation, nullptr if sequence complete
     */
    BatchOperation* load_next_batch();
    
    /**
     * Load specific batch by index
     * @param index Batch index to load
     * @return Pointer to loaded batch operation, nullptr if invalid index
     */
    BatchOperation* load_batch(size_t index);
    
    /**
     * Check if more batches remain in sequence
     * @return true if more batches available
     */
    bool has_next_batch() const { return current_batch_index < operation_sequence.size(); }
    
    /**
     * Get current batch progress information
     * @return pair<current_index, total_batches>
     */
    pair<size_t, size_t> get_progress() const { 
        return make_pair(current_batch_index, operation_sequence.size()); 
    }
    

    
    // === File Operations ===
    
    /**
     * Parse batch filename to extract index and operation type
     * @param filename Batch filename (e.g., "05_inc_batch.mtx")
     * @param index Output: extracted index
     * @param op_type Output: extracted operation type
     * @return true if parsing successful
     */
    bool parse_batch_filename(const string& filename, int& index, OperationType& op_type);
    
    /**
     * Load edges from batch file into GPU-ready format
     * @param batch Batch operation to populate with edge data
     * @return true if loading successful
     */
    bool load_batch_edges(BatchOperation& batch);
    
    /**
     * Count edges in batch file
     * @param filename Path to batch file
     * @return Number of edges in file
     */
    size_t count_edges_in_file(const string& filename);
    
    // === Utility Functions ===
    
    /**
     * Print current stream statistics
     */
    void print_statistics() const;
    
    /**
     * Validate batch file format and accessibility
     * @param filename Path to batch file
     * @return true if file is valid and accessible
     */
    bool validate_batch_file(const string& filename);
};

// === Standalone Functions for Compatibility ===


GraphInfo auto_detect_graph_info(const char* filename);

#endif
/**
 * Implementation file for dyGRASS Random Walk Decremental Graph Sparsification
 *
 * This file implements the core functionality for decremental graph sparsification:
 * - Enhanced CSR graph construction with edge mapping for efficient deletion
 * - Target edge loading for batch processing
 * - Edge removal operations based on random walk results
 * 
 * Key Differences from Incremental Version:
 * - Builds comprehensive edge mapping during construction for O(1) deletion
 * - Processes edges for removal rather than addition
 * - Maintains connectivity by preserving edges when no alternative paths exist
 * 
 * Memory-Mapped I/O Approach:
 * - Uses mmap() for efficient large file processing
 * - Multi-pass parsing for memory efficiency
 * - Specialized data layouts for GPU-friendly access patterns
 */

#include <iostream>
#include <tuple>
#include <fstream>
#include <cmath>  // for pow() in multiplier calculation
#include <sys/stat.h> // for stat - file size information
#include <sys/mman.h> // for mmap - memory mapping files
#include <fcntl.h> // for open - file descriptor operations
#include <assert.h> // for assert - debug assertions
#include <vector>
#include <unistd.h> // for close - closing file descriptors
#include "functions.h"
using namespace std;



/**
 * Get file size efficiently using system stat call
 * @param filename Path to the file
 * @return File size in bytes, or -1 on error
 */
inline off_t fsize(const char *filename) {
    struct stat st; // stat structure to store file information
    if (stat(filename, &st) == 0) // read file information to st, 0 means success
        return st.st_size;
    return -1; // Return -1 if file doesn't exist or can't be accessed
}



/**
 * CSRGraph Constructor: Load graph from file with enhanced edge mapping
 * 
 * Enhanced 3-pass construction algorithm:
 * 1. Count edges and determine vertex range
 * 2. Calculate vertex degrees and setup CSR structure
 * 3. Populate adjacency lists AND build edge mapping for O(1) deletion
 * 
 * Key Enhancement for Decremental Processing:
 * - Builds unordered_map during construction for fast edge lookup
 * - Edge mapping: (src,dest) -> (position_in_src_adj, position_in_dest_adj)
 * - Uses multiplier-based hash keys for undirected edge pairs
 * 
 * @param filename Path to graph file (MTX format expected)
 * @param is_reverse Create undirected graph (add reverse edges)
 * @param skip_head Number of header lines to skip
 * @param weightFlag 1 if file contains weights, 0 for unweighted
 * 
 * NOTE: Input graph is expected to be 1-based indexed and converted to 0-based!
 */
CSRGraph::CSRGraph(const char* filename, bool is_reverse, long skip_head, int weightFlag) {
    // === File I/O Setup with Memory Mapping ===
    int fd = open(filename, O_RDONLY);  // Open file in read-only mode
    if (fd == -1) {
        perror("Error opening file");
        exit(-1);
    }

    size_t file_size = fsize(filename);  // Get the file size
    // Memory map entire file for efficient access (avoids repeated read() calls)
    char* ss_head = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(ss_head != MAP_FAILED);  // Check if mmap succeeded
    madvise(ss_head, file_size, MADV_SEQUENTIAL);  // Hint OS for sequential access optimization

    // Skip header lines (e.g., MTX format headers like "%%MatrixMarket matrix coordinate...")
    size_t head_offset = 0;
    int skip_count = 0;
    while (skip_count < skip_head && head_offset < file_size) {
        if (ss_head[head_offset++] == '\n') skip_count++;
    }

    char* ss = ss_head + head_offset;  // Point to actual data after headers
    file_size -= head_offset;  // Adjust file size after skipping headers
    size_t curr = 0, next = 0, edge_count = 0;  // Parse positions and edge counter
    vertex_t v_max = 0, v_min = INFTY;  // Track vertex ID range
    int checkwt = 0;  // Counter for weight field parsing (src, dest, weight = 3 fields)
    
    // === PASS 1: Count edges and determine vertex range ===
    // This pass scans the entire file to:
    // 1. Count total number of edges (for memory allocation)
    // 2. Find min/max vertex IDs (to determine graph size)
    while (next < file_size) {
        // Parse source vertex and convert from 1-based to 0-based
        vertex_t src = atoi(ss + curr) - 1;
        if (weightFlag) checkwt++;  // Track which field we're parsing (1=src, 2=dest, 3=weight)
        if (checkwt != 3) {         // Only update vertex bounds for src/dest, not weight
            v_max = max(v_max, src);
            v_min = min(v_min, src);
        }
        // Skip to next field (space/tab/newline delimited)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        // Note: We don't need to parse dest in pass 1, just count
        if (checkwt != 3) edge_count++;  // Count edges (but not weight entries)
        if (checkwt == 3) checkwt = 0;  // Reset weight counter after processing weight
    }

    // === Initialize Enhanced CSR Graph Structure ===
    
    this->line_count = edge_count>>1;  // Divide by 2 since each line has src->dest pair
    // Edge count depends on whether we're creating undirected graph (with reverse edges)
    this->edge_count = is_reverse ? edge_count : edge_count>>1;
    this->vert_count = v_max + 1;  // Vertex count based on max ID (0-indexed)
    this->v_max = v_max;
    this->v_min = v_min;
    
    // Calculate multiplier for edge mapping hash keys
    // This ensures (src * multiplier + dest) produces unique keys for all vertex pairs
    int digit_num = findDigitsNum(vert_count);
    this->multiplier = pow(10, digit_num);

    // Allocate memory for enhanced CSR data structures
    this->begin.resize(this->vert_count + 1, 0);  // CSR begin array (vert_count+1 for easier indexing)
    this->adj.resize(this->edge_count);           // Adjacency list (all neighbors concatenated)
    this->weight.resize(this->edge_count);        // Edge weights corresponding to adj entries
    this->degree.resize(this->vert_count, 0);     // Vertex degrees (will be recalculated)
    this->from.resize(this->edge_count);          // Original edge index mapping
    this->mtx.resize(line_count);                 // Original edge tuples (src, dest, weight)
    this->reverse.resize(this->edge_count);       // Reverse edge mapping
    
    // === Key Enhancement: Edge Mapping for O(1) Deletion ===
    // Reserve space for edge mapping hash table (50% extra for load factor optimization)
    this->edge_map.reserve((unsigned)(this->line_count)*1.5);

    // === PASS 2: Calculate vertex degrees ===
    // Second pass through file to count outgoing edges per vertex
    // This is needed to construct the CSR begin array
    curr = next = 0;
    while (next < file_size) {
        vertex_t src = atoi(ss + curr) - 1;  // Convert from 1-based to 0-based
        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        curr = next;
        vertex_t dest = atoi(ss + curr) - 1;  // Convert from 1-based to 0-based
        
        // Skip to weight field (if present)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        if(weightFlag != 0) { // Skip weight value if present in file
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }

        curr = next;
        // Increment degree counters
        this->degree[src]++;  // Source vertex gains an outgoing edge
        if (is_reverse) this->degree[dest]++;  // If undirected, dest also gains outgoing edge
    }
    
    // Convert degrees to CSR begin array using cumulative sum
    // begin[i] = starting index in adj[] array for vertex i's neighbors
    this->begin[0] = 0;
    for (size_t i = 1; i <= this->vert_count; i++) {
        this->begin[i] = this->begin[i - 1] + this->degree[i - 1];
    }

    // Reset degree arrays to use as insertion counters in Pass 3
    std::fill(this->degree.begin(), this->degree.end(), 0);

    // === PASS 3: Populate adjacency lists AND build edge mapping ===
    // Third pass fills the actual CSR adjacency lists using the begin array
    // AND constructs the critical edge mapping for O(1) deletion
    curr = next = 0;
    size_t offset = 0;  // Tracks original edge index in file
    while (offset < line_count) { 
        vertex_t src = atoi(ss + curr) - 1;  // Convert from 1-based to 0-based
        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        vertex_t dest = atoi(ss + curr) - 1;  // Convert from 1-based to 0-based
        
        // Insert edge src->dest into CSR adjacency list
        // Position: begin[src] + current_degree[src] gives next available slot
        index_t pos_src = this->begin[src] + this->degree[src];
        this->adj[pos_src] = dest;              // Store destination vertex
        this->from[pos_src] = offset;           // Map back to original edge index
        this->reverse[pos_src] = this->degree[dest];  // Store current degree of destination
        
        // Handle reverse edges for undirected graphs
        if (is_reverse) {
            if (dest == src) {
                // Self-loop: need special handling to avoid double-counting
                index_t pos_dest = this->begin[dest] + this->degree[dest] + 1;
                this->adj[pos_dest] = src;
                this->from[pos_dest] = offset;
                this->reverse[pos_dest] = this->degree[src];
            }
            else {
                // Regular reverse edge dest->src
                index_t pos_dest = this->begin[dest] + this->degree[dest];
                this->adj[pos_dest] = src;
                this->from[pos_dest] = offset;
                this->reverse[pos_dest] = this->degree[src];
            }
        }


        // Parse and assign edge weights
        weight_t wtvalue;
        if (weightFlag != 0) {
            // Skip to weight field and parse floating point value
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
            curr = next;
            wtvalue = atof(ss + curr);  // Parse weight as float
        }
        else {
            wtvalue = 1.0;  // Default weight for unweighted graphs
        }

        // Store weights in CSR weight array
        this->weight[this->begin[src] + this->degree[src]] = wtvalue;
        if (is_reverse) this->weight[this->begin[dest] + this->degree[dest]] = wtvalue;

        // Store original edge tuple for reference
        this->mtx[offset] = make_tuple(src, dest, wtvalue);

        // === BUILD EDGE MAPPING FOR O(1) DELETION ===
        // Create unique hash key for undirected edge pair
        long a = src, b = dest;
        int degree_a = this->degree[a], degree_b = this->degree[b];
        if (a > b) {
            swap(a, b);           // Ensure consistent ordering (smaller vertex first)
            swap(degree_a, degree_b);   
        } 
        
        // Generate unique key: smaller_vertex * multiplier + larger_vertex
        long key = a * this->multiplier + b;
        
        // Map to current positions in adjacency arrays
        // This enables O(1) lookup for edge deletion: given (src,dest), find positions immediately
        pair<index_t,index_t> value = {degree_a, degree_b};
        this->edge_map.insert({key, value});

        // Advance to next line for next iteration
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        
        // Update degree counters (used as insertion positions)
        this->degree[src]++;
        if (is_reverse) this->degree[dest]++;
        
        offset++;  // Move to next edge
    }
    
    // Clean up memory mapping and file descriptor
    munmap(ss_head, file_size);
    close(fd);

    // Set up raw pointers for GPU transfer (pointing to vector data)
    this->adj_list = this->adj.data();
    this->weight_list = this->weight.data();
    this->beg_pos = this->begin.data();
    this->degree_list = this->degree.data();
}

/**
 * Remove edges from graph based on random walk results
 * 
 * Core decremental sparsification operation:
 * - Removes edges when random walks find alternative connecting paths
 * - Preserves edges when no alternative paths exist (maintains connectivity)
 * - Uses edge mapping for efficient O(1) edge deletion
 * 
 * This method is called after random walk processing to update
 * the graph structure based on sparsification results.
 * 
 * TODO: Implement edge removal logic based on random walk path finding results
 * Currently placeholder - actual implementation depends on random walk results format
 */
void CSRGraph::remove_edge_from_targets(){
    // Implementation pending - will process random walk results and
    // remove edges where alternative paths were found
}



/**
 * Read target edges for decremental processing with batch organization
 * 
 * Loads edges that are candidates for removal during decremental sparsification.
 * These edges will be tested with random walks to determine if they can be
 * safely removed while preserving graph connectivity.
 * 
 * Memory Layout (same as incremental but for removal):
 * - First half: source vertices of edges to potentially remove
 * - Second half: destination vertices of edges to potentially remove
 * - Organized in batches for GPU processing efficiency
 * 
 * @param filename Path to target edges file
 * @param process_line Number of edges to process (-1 for all)
 * @param weightFlag 1 if file contains weights, 0 for unweighted
 * @param divisor Batch division factor (typically 10 for 10 experimental batches)
 * 
 * NOTE: Input edges are expected to be 1-based and converted to 0-based!
 */
void CSRGraph::read_targets(const char* filename, int process_line, int weightFlag, int divisor) {

    this->divisor = divisor;

    // === Memory-mapped file setup ===
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("File open error");
        exit(-1);
    }

    size_t file_size = fsize(filename);
    char* ss_head = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(ss_head != MAP_FAILED);
    madvise(ss_head, file_size, MADV_SEQUENTIAL);

    char* ss = ss_head;
    size_t curr = 0, next = 0;
    size_t edge_count = 0;
    int checkwt = 0;

    // === PASS 1: Count total edges in file ===
    while (next < file_size) {
        if (weightFlag != 0) checkwt++;  // Track field position for weight parsing

        // Skip current field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        if (checkwt != 3) edge_count++;  // Count edges (ignore weight fields)
        if (checkwt == 3) checkwt = 0;   // Reset after processing weight
    }

    const index_t line_count = edge_count >> 1;  // Divide by 2 since we count src+dest separately
    edge_count = edge_count >> 1;
    if (process_line != -1) {
        edge_count = process_line;  // Limit processing if specified
    }

    // === Memory allocation for batch-organized layout ===
    // Layout: [src_batch_1, src_batch_2, ..., src_batch_N, 
    //          dest_batch_1, dest_batch_2, ..., dest_batch_N]
    index_t* ext = new index_t[edge_count * 2];  // Double size: half for src, half for dest
    weight_t* weights = nullptr;
    if (weightFlag != 0) {
        weights = new weight_t[edge_count]; // Weights stored separately
    }
    
    // === PASS 2: Parse and arrange edges in batch layout ===
    vertex_t source, dest;
    size_t offset = 0;  // Current edge being processed
    curr = 0;
    next = 0;

    // Batch organization: divide edges into 'divisor' batches for processing
    int interval = (edge_count + divisor - 1) / divisor;  // Ceiling division for batch size
    int initial_interval = interval;
    weight_t wtvalue;
    int write_location = 0;  // Current write position in the ext array
    while (offset < edge_count) {
        // Parse source vertex (convert from 1-based to 0-based)
        char* sss = ss + curr;
        source = atoi(sss) - 1; // Convert from 1-based to 0-based indexing

        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        
        // Parse destination vertex (convert from 1-based to 0-based)
        char* sss1 = ss + curr;
        dest = atoi(sss1) - 1; // Convert from 1-based to 0-based indexing

        // Skip to weight field (if present)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        
        // Parse weight if present
        if (weightFlag != 0) {
            curr = next;
            char* sss1 = ss + curr;
            wtvalue = atof(sss1);
            weights[offset] = wtvalue;

            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }
        curr = next;

        // Store in specialized batch layout: source in first half, destination offset by interval
        // This creates: [src_batch1, src_batch2, ..., src_batchN, dest_batch1, dest_batch2, ..., dest_batchN]
        ext[write_location] = source;               // Source in current batch
        ext[write_location + interval] = dest;      // Destination in corresponding position

        offset++;
        write_location++;
        
        // Move to next batch when current batch is full
        if ((write_location % interval == 0) && (interval == initial_interval)) {
            write_location += interval;  // Jump to next batch start
            
            // Handle last batch which might be smaller
            if (write_location + interval * 2 > edge_count * 2) {
                interval = edge_count - write_location / 2;
            }
        }
    }

    // Clean up memory mapping and file descriptor
    munmap(ss_head, file_size);
    close(fd);

    // Assign parsed data to class members for GPU processing
    this->sample_weights = weights;  // Edge weights (may be nullptr if unweighted)
    this->sample_nodes = ext;        // Edge endpoints in batch layout
    this->sample_count = edge_count; // Number of target edges to process
}





/**
 * Graph I/O Functions for dyGRASS Random Walk Incremental Processing
 * 
 * This file implements efficient graph reading and parsing functions using:
 * - Memory-mapped I/O for large file processing
 * - Compressed Sparse Row (CSR) format for graph storage
 * - Multi-pass parsing for memory efficiency
 */

#include<iostream>
#include<tuple>
#include<fstream>
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
 * Read graph from file and construct Compressed Sparse Row (CSR) representation
 * 
 * CSR Format stores a graph as:
 * - adj[]: adjacency list (all neighbors concatenated)
 * - begin[]: starting index in adj[] for each vertex
 * - degree[]: number of neighbors for each vertex
 * - weight[]: edge weights corresponding to adj[] entries
 * 
 * Uses 3-pass algorithm for memory efficiency:
 * 1. Count edges and find vertex range
 * 2. Calculate vertex degrees
 * 3. Populate adjacency lists
 * 
 * @param filename Path to graph file (MTX format expected)
 * @param is_reverse If true, create undirected graph (add reverse edges)
 * @param skip_head Number of header lines to skip
 * @param weightFlag 1 if file contains weights, 0 for unweighted
 * @return CSRGraph structure containing the parsed graph
 * 
 * NOTE: Input graph is expected to be 0-based indexed!
 */
CSRGraph read_graph(const char* filename, bool is_reverse, long skip_head, int weightFlag) {
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
        // Parse source vertex
        vertex_t src = atoi(ss + curr);
        if (weightFlag) checkwt++;  // Track which field we're parsing (1=src, 2=dest, 3=weight)
        if (checkwt != 3) {         // Only update vertex bounds for src/dest, not weight
            v_max = max(v_max, src);
            v_min = min(v_min, src);
        }
        // Skip to next field (space/tab/newline delimited)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        // Parse destination vertex
        vertex_t dest = atoi(ss + curr);
        if (checkwt != 3) edge_count++;  // Count edges (but not weight entries)
        if (checkwt == 3) checkwt = 0;  // Reset weight counter after processing weight
    }

    // === Initialize CSR Graph Structure ===
    CSRGraph graph;
    const index_t line_count = edge_count>>1;  // Divide by 2 since each line has src->dest pair
    // Edge count depends on whether we're creating undirected graph (with reverse edges)
    graph.edge_count = is_reverse ? edge_count : edge_count>>1;
    graph.vert_count = v_max + 1;  // Vertex count based on max ID (0-indexed)
    graph.v_max = v_max;
    graph.v_min = v_min;

    // Allocate memory for CSR data structures
    graph.begin.resize(graph.vert_count + 1, 0);  // CSR begin array (vert_count+1 for easier indexing)
    graph.adj.resize(graph.edge_count);           // Adjacency list (all neighbors concatenated)
    graph.weight.resize(graph.edge_count);        // Edge weights corresponding to adj entries
    graph.degree.resize(graph.vert_count, 0);     // Vertex degrees (will be recalculated)
    graph.from.resize(graph.edge_count);          // Original edge index mapping
    graph.mtx.resize(line_count);                 // Original edge tuples (src, dest, weight)
    graph.reverse.resize(graph.edge_count);       // Reverse edge mapping

    // === PASS 2: Calculate vertex degrees ===
    // Second pass through file to count outgoing edges per vertex
    // This is needed to construct the CSR begin array
    curr = next = 0;
    while (next < file_size) {
        vertex_t src = atoi(ss + curr);
        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        curr = next;
        vertex_t dest = atoi(ss + curr);
        
        // Skip to weight field (if present)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        if(weightFlag != 0) { // Skip weight value if present in file
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }

        curr = next;
        // Increment degree counters
        graph.degree[src]++;  // Source vertex gains an outgoing edge
        if (is_reverse) graph.degree[dest]++;  // If undirected, dest also gains outgoing edge
    }
    
    // Convert degrees to CSR begin array using cumulative sum
    // begin[i] = starting index in adj[] array for vertex i's neighbors
    graph.begin[0] = 0;
    for (size_t i = 1; i <= graph.vert_count; i++) {
        graph.begin[i] = graph.begin[i - 1] + graph.degree[i - 1];
    }

    // Reset degree arrays to use as insertion counters in Pass 3
    std::fill(graph.degree.begin(), graph.degree.end(), 0);

    // === PASS 3: Populate adjacency lists and edge data ===
    // Third pass fills the actual CSR adjacency lists using the begin array
    curr = next = 0;
    size_t offset = 0;  // Tracks original edge index in file
    while (offset < line_count) { 
        vertex_t src = atoi(ss + curr);
        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        vertex_t dest = atoi(ss + curr);
        
        // Insert edge src->dest into CSR adjacency list
        // Position: begin[src] + current_degree[src] gives next available slot
        index_t pos = graph.begin[src] + graph.degree[src];
        graph.adj[pos] = dest;              // Store destination vertex
        graph.from[pos] = offset;           // Map back to original edge index
        graph.reverse[pos] = graph.degree[dest];  // Store current degree of destination
        
        // Handle reverse edges for undirected graphs
        if (is_reverse) {
            if (dest == src) {
                // Self-loop: need special handling to avoid double-counting
                graph.adj[graph.begin[dest] + graph.degree[dest] + 1] = src;
                graph.from[graph.begin[dest] + graph.degree[dest] + 1] = offset;
                graph.reverse[graph.begin[dest] + graph.degree[dest] + 1] = graph.degree[src];
            }
            else {
                // Regular reverse edge dest->src
                index_t rev_pos = graph.begin[dest] + graph.degree[dest];
                graph.adj[rev_pos] = src;
                graph.from[rev_pos] = offset;
                graph.reverse[rev_pos] = graph.degree[src];
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
        graph.weight[graph.begin[src] + graph.degree[src]] = wtvalue;
        if (is_reverse) graph.weight[graph.begin[dest] + graph.degree[dest]] = wtvalue;

        // Store original edge tuple for reference
        graph.mtx[offset] = make_tuple(src, dest, wtvalue);

        // Advance to next line for next iteration
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        
        // Update degree counters (used as insertion positions)
        graph.degree[src]++;
        if (is_reverse) graph.degree[dest]++;
        
        offset++;  // Move to next edge
    }

    // Clean up memory mapping and file descriptor
    munmap(ss_head, file_size);
    close(fd);

    return graph;  // Return completed CSR graph structure
}


/**
 * Read incremental edge updates (target edges) for dynamic graph experiments
 * 
 * This function reads edge extension files that contain new edges to be added
 * to the graph incrementally. The edges are organized in a special layout for
 * experimental purposes - divided into 10 segments for multiple trial runs.
 * 
 * Memory Layout:
 * - First half: source vertices for all edges
 * - Second half: destination vertices for all edges
 * - Organized in 10 segments to support 10 experimental trials
 * 
 * @param filename Path to extension/target edge file
 * @param process_line Number of edges to process (-1 for all)
 * @param weightFlag 1 if file contains weights, 0 for unweighted
 * @return Targets structure containing edge arrays and count
 * 
 * NOTE: Input edges are expected to be 1-based and converted to 0-based!
 */
Targets read_targets(const char* filename, int process_line, int weightFlag) {
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

    // === Memory allocation for special 10-segment layout ===
    // Layout: [src_segment_1, src_segment_2, ..., src_segment_10, 
    //          dest_segment_1, dest_segment_2, ..., dest_segment_10]
    index_t* ext = new index_t[edge_count * 2];  // Double size: half for src, half for dest
    weight_t* weights = new weight_t[edge_count]; // Weights stored separately
    // === PASS 2: Parse and arrange edges in 10-segment layout ===
    vertex_t source, dest;
    size_t offset = 0;  // Current edge being processed
    curr = 0;
    next = 0;

    // Experimental setup: divide edges into 10 segments for multiple trials
    int divisor = 10;
    int interval = (edge_count + divisor - 1) / divisor;  // Ceiling division for segment size
    int initial_interval = interval;
    weight_t wtvalue;
    int write_location = 0;  // Current write position in the ext array
    
    while (offset < edge_count) {
        // Parse source vertex (convert from 1-based to 0-based)
        char* sss = ss + curr;
        source = atoi(sss) - 1; 

        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        
        // Parse destination vertex (convert from 1-based to 0-based)
        char* sss1 = ss + curr;
        dest = atoi(sss1) - 1; 

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

        // Store in special layout: source in first half, destination offset by interval
        // This creates: [src_seg1, src_seg2, ..., src_seg10, dest_seg1, dest_seg2, ..., dest_seg10]
        ext[write_location] = source;               // Source in current segment
        ext[write_location + interval] = dest;      // Destination in corresponding position

        offset++;
        write_location++;
        
        // Move to next segment when current segment is full
        if ((write_location % interval == 0) && (interval == initial_interval))  {
            write_location += interval;  // Jump to next segment start
            // Handle last segment which might be smaller
            if (write_location + interval * 2 > edge_count * 2) {
                interval = edge_count - write_location / 2;
            }
        }
    }

    // Clean up memory mapping and file descriptor
    munmap(ss_head, file_size);
    close(fd);

    // Return structure containing edge arrays and count
    return {ext, weights, edge_count};
}




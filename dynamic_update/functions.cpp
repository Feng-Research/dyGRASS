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
#include <cstring>
#include <dirent.h> // for directory operations
#include <algorithm> // for sort
#include <regex> // for filename parsing
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

GraphInfo auto_detect_graph_info(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
      cout << "Error: Cannot open file '" << filename << "' for reading. "
           << "Reason: " << strerror(errno) << endl;
      exit(-1);
    }


    size_t file_size = fsize(filename);
    char* ss_head = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(ss_head != MAP_FAILED);
    madvise(ss_head, file_size, MADV_SEQUENTIAL);

    char* ss = ss_head;
    size_t curr = 0, next = 0;

    vertex_t v_max = 0, v_min = INFTY;  // Track vertex ID range
    bool has_matrixmarket_header = false;
    bool has_dimensions_line = false;
    bool is_weighted = false;
    bool is_laplacian = false;
    int base;
    long skip_lines = 0;

    // find if the header part exists
    while (next < file_size){
        if (ss_head[next] == '%') {

            has_matrixmarket_header = true;
            skip_lines++;
            while (ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == '\n' || ss[next] == '\t') next++;
        }
        else{
            break;
        }
    }
    // find if the dimensions line exists
    // assume the dimension line is <rows> <cols> <nonzeros>
    // Parse the first non-comment line
    curr = next; // Now curr record the first non-comment line
    char* line_start = ss + curr;
    while (ss[next] != '\n' && ss[next] != '\t') next++;
    while (ss[next] == '\n' || ss[next] == '\t') next++;

    // Extract three numbers
    double num1, num2, num3;
    double dimensions = 0;
    
    // Create reusable buffer for line parsing
    const size_t line_buffer_size = 256;
    char* line_buffer = new char[line_buffer_size];
    
    // Find the end of current line to avoid sscanf reading ahead
    size_t line_end = curr;
    while (line_end < file_size && ss[line_end] != '\n') line_end++;
    
    // Create a null-terminated string for this line only
    size_t line_length = line_end - curr;
    int parsed = 0;
    if (line_length < line_buffer_size - 1) {
        strncpy(line_buffer, line_start, line_length);
        line_buffer[line_length] = '\0';
        
        parsed = sscanf(line_buffer, "%lf %lf %lf", &num1, &num2, &num3);
    }

    if (parsed == 3) {
        // Check if third number is integer (dimensions) or float (edge weight)
        if (num3 == (long)num3 && num3 > 1 && num1 == num2) {
            // num3 is integer - could be dimensions line

            if (num1 * num2 > num3 && num3 >= num1 -1){
                has_dimensions_line = true; // This is tentatively a dimensions line
                dimensions = num1;
                skip_lines++;
            }
        } 
    }

    if (!has_dimensions_line) {
        v_max = max(v_max, (vertex_t)num1);
        v_min = min(v_min, (vertex_t)num1);

        v_max = max(v_max, (vertex_t)num2);
        v_min = min(v_min, (vertex_t)num2);
    }


    // check weight
    // check laplacian
    // check base
    // check if its a triangle matrix or full matrix
    bool is_triangle = true;
    bool checked_triangle = false;

    double edge_first, edge_second;
    bool check_edge_loaded = false;
    // check if weighted or not

    curr = next;
    line_start = ss + curr;
    while (ss[next] != '\n' && ss[next] != '\t') next++;
    while (ss[next] == '\n' || ss[next] == '\t') next++;

    // Parse second data line with proper line termination  
    line_end = curr;
    while (line_end < file_size && ss[line_end] != '\n') line_end++;
    line_length = line_end - curr;
    if (line_length < line_buffer_size - 1) {
        strncpy(line_buffer, line_start, line_length);
        line_buffer[line_length] = '\0';
        
        parsed = sscanf(line_buffer, "%lf %lf %lf", &num1, &num2, &num3);
    } else {
        parsed = 0; // Skip overly long lines
    }
    if (parsed == 3) {
        // cout << "check point 3" << endl;
        // cout << "num1: " << num1 << ", num2: " << num2 << ", num3: " << num3 << endl;

        is_weighted = true;

        if(num1 != num2){
            edge_first = max(num1, num2);
            edge_second = min(num1, num2);
            check_edge_loaded = true;
        }
    }
    else if (parsed == 2) {
        // cout << "check point 4" << endl;
        // cout << "num1: " << num1 << ", num2: " << num2 << endl;
        is_weighted = false;
        is_laplacian = false;
    }
    else {
        // Invalid line format, quit program
        cerr << "Invalid line format, quitting program." << endl;
        exit(-1);
    }


    v_max = max(v_max, (vertex_t)num1);
    v_min = min(v_min, (vertex_t)num1);

    v_max = max(v_max, (vertex_t)num2);
    v_min = min(v_min, (vertex_t)num2);


    
    while(next < file_size){
        curr = next;
        
        // Fast forward to next number (skip whitespace)
        while (curr < file_size && (ss[curr] == ' ' || ss[curr] == '\t')) curr++;
        if (curr >= file_size) break;
        
        // Quick extraction of first two numbers without full line parsing
        double fast_num1 = 0, fast_num2 = 0;
        char* endptr1;
        char* endptr2;
        
        fast_num1 = strtod(ss + curr, &endptr1);
        if (endptr1 > ss + curr) {
            fast_num2 = strtod(endptr1, &endptr2);
            if (endptr2 > endptr1) {
                // Successfully parsed two numbers - update min/max
                v_max = max(v_max, (vertex_t)fast_num1);
                v_min = min(v_min, (vertex_t)fast_num1);
                v_max = max(v_max, (vertex_t)fast_num2);
                v_min = min(v_min, (vertex_t)fast_num2);
            }
        }
        

            // Find line end for detailed parsing
            size_t line_end = curr;
            while (line_end < file_size && ss[line_end] != '\n') line_end++;
            
            size_t line_length = line_end - curr;
            if (line_length > 0 && line_length < line_buffer_size - 1) {
                strncpy(line_buffer, ss + curr, line_length);
                line_buffer[line_length] = '\0';
                
                if (is_weighted) {
                    parsed = sscanf(line_buffer, "%lf %lf %lf", &num1, &num2, &num3);
                    if (parsed >= 3 && num3 < 0) {
                        is_laplacian = true; // Negative weights indicate Laplacian matrix
                    }
                } else {
                    parsed = sscanf(line_buffer, "%lf %lf", &num1, &num2);
                }
                
                if (parsed >= 2) {
                    if (!check_edge_loaded && num1 != num2) {
                        edge_first = max(num1, num2);
                        edge_second = min(num1, num2);
                        check_edge_loaded = true;
                    } else if (!checked_triangle && check_edge_loaded) {
                        // Check if this is a triangle matrix
                        if (max(num1,num2) == edge_first && min(num1,num2) == edge_second) {
                            is_triangle = false; 
                            checked_triangle = true; // Only check once
                        }
                    }
                }
            }

        
        // Move to next line (fast)
        while (next < file_size && ss[next] != '\n') next++;
        next++; // Skip the newline
        while (next < file_size && ss[next] == '\n') next++; // Skip multiple newlines
    }

    // Clean up reusable buffer
    delete[] line_buffer;

    if(v_min == 1){
        base = 1; // 1-based indexing
    } else if (v_min == 0) {
        base = 0; // 0-based indexing
    } else {
        cerr << "Error: Vertex IDs must be either 0 or 1-based." << endl;
        exit(-1);
    }

    // Validate dimensions now that we've scanned the entire file
    if (base == 1){
        if (has_dimensions_line && v_max != dimensions){
            cout << "Warning: Max vertex ID (" << v_max << ") does not match dimensions (" << dimensions << ")." << endl;
            has_dimensions_line = false; // No dimensions line if max vertex ID doesn't match
        }
    } else {
        if (has_dimensions_line && v_max + 1 != dimensions){
            cout << "Warning: Max vertex ID (" << v_max << ") does not match dimensions (" << dimensions << ")." << endl;
            has_dimensions_line = false; // No dimensions line if max vertex ID doesn't match
        }
    }


    munmap(ss_head, file_size);
    close(fd);

    return {
        skip_lines,
        has_matrixmarket_header,
        has_dimensions_line,
        is_weighted,
        is_laplacian,
        is_triangle,
        base,
        v_max, 
        v_min
    };
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
 */
CSRGraph::CSRGraph(const char* filename) {

    GraphInfo info = auto_detect_graph_info(filename);
    long skip_head = info.skip_lines;
    bool weightFlag = info.is_weighted;
    bool is_reverse = info.is_triangle;
    bool is_laplacian = info.is_laplacian;
    int base = info.base;
    vertex_t v_max_prev = info.v_max;
    vertex_t v_min_prev = info.v_min;

    // === File I/O Setup with Memory Mapping ===
    int fd = open(filename, O_RDONLY);  // Open file in read-only mode
    if (fd == -1) {
      cout << "Error: Cannot open file '" << filename << "' for reading. "
           << "Reason: " << strerror(errno) << endl;
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
        
        this->line_count++;
        // Parse source vertex and convert to 0-based
        vertex_t src = atoi(ss + curr) - base;
        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        curr = next;
        vertex_t dest = atoi(ss + curr) - base;  // Convert from 1-based to 0-based
        
        // Skip to weight field (if present)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;


        if(weightFlag != 0) { // Skip weight value if present in file
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }
        curr = next;

        if (is_laplacian && src == dest) {
            // Skip diagonal in Laplacian matrix
            continue;
        }
        v_max = max(v_max, src);
        v_min = min(v_min, src);

        v_max = max(v_max, dest);
        v_min = min(v_min, dest);
        this->edge_count++;
        
    }

    // === Initialize Enhanced CSR Graph Structure ===
    
    // Edge count depends on whether we're creating undirected graph (with reverse edges)
    this->edge_count = is_reverse ? edge_count << 1 : edge_count;
    this->vert_count = v_max + 1 - base; // auto determined by base
    this->v_max = v_max;
    this->v_min = v_min;

    assert(v_max == v_max_prev && v_min == v_min_prev);
    assert(v_min == base);
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
    // this->mtx.resize(line_count);                 // Original edge tuples (src, dest, weight)
    this->reverse.resize(this->edge_count);       // Reverse edge mapping
    
    // === Key Enhancement: Edge Mapping for O(1) Deletion ===
    // Reserve space for edge mapping hash table (50% extra for load factor optimization)
    this->edge_map.reserve((unsigned)(this->line_count)*1.5);

    // === PASS 2: Calculate vertex degrees ===
    curr = next = 0;
    while (next < file_size) {
        vertex_t src = atoi(ss + curr) - base;  // Convert to 0 based 
        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        curr = next;
        vertex_t dest = atoi(ss + curr) - base;  // Convert to 0 based
        
        // Skip to weight field (if present)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        if(weightFlag != 0) { // Skip weight value if present in file
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }

        curr = next;
        // Increment degree counters
        if (is_laplacian && src == dest) {
            // Skip diagonal in Laplacian matrix
            continue;
        }
        this->degree[src]++;  // Source vertex gains an outgoing edge
        if (is_reverse) this->degree[dest]++;  // If triangle, dest also gains outgoing edge
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
        vertex_t src = atoi(ss + curr) - base;  // Convert to 0-based
        // Skip to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        vertex_t dest = atoi(ss + curr) - base;  // Convert to 0-based

        // Parse and assign edge weights
        weight_t wtvalue;
        if (weightFlag) {
            // Skip to weight field and parse floating point value
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
            curr = next;
            wtvalue = atof(ss + curr);  // Parse weight as float
            
            // For Laplacian matrices, off-diagonal entries are negative
            // Convert to positive edge weights for graph algorithms
            if (is_laplacian) {
                wtvalue = -wtvalue;  // Make negative weights positive
            }
        }
        else {
            wtvalue = 1.0;  // Default weight for unweighted graphs
        }

        // Advance to next line for next iteration
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        
        if (is_laplacian && src == dest) {
            // Skip diagonal in Laplacian matrix
            continue;
        }
        // Insert edge src->dest into CSR adjacency list
        // Position: begin[src] + current_degree[src] gives next available slot
        index_t pos_src = this->begin[src] + this->degree[src];
        this->adj[pos_src] = dest;              // Store destination vertex
        this->from[pos_src] = offset;           // Map back to original edge index
        this->reverse[pos_src] = this->degree[dest];  // Store current degree of destination

        // Handle reverse edges for triangular matrix
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

        // Store weights in CSR weight array
        this->weight[this->begin[src] + this->degree[src]] = wtvalue;
        if (is_reverse) this->weight[this->begin[dest] + this->degree[dest]] = wtvalue;

        // Store original edge tuple for reference
        // this->mtx[offset] = make_tuple(src, dest, wtvalue);

        // === BUILD EDGE MAPPING FOR O(1) DELETION ===
        // Create unique hash key for undirected edge pair
        long a = src, b = dest;
        index_t degree_a = this->degree[a], degree_b = this->degree[b];
        if (a > b) {
            swap(a, b);           // Ensure consistent ordering (smaller vertex first)
            swap(degree_a, degree_b);   
        } 
        
        // Generate unique key: smaller_vertex * multiplier + larger_vertex
        long key = a * this->multiplier + b;
        
        // Map to current positions in adjacency arrays
        // This enables O(1) lookup for edge deletion: given (src,dest), find positions immediately


        this->edge_map.emplace(key, EdgeInfo{degree_a, degree_b, wtvalue});

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

EdgeStream::EdgeStream(const string& folder, int base) {
    this->base = base;
    this->operations_folder = folder;
    this->batch_index = -1;
    this->batch_size = 0;
    this->current_op = OperationType::INCREMENTAL;
    this->total_edges_processed = 0;
    this->manual_selection = true;
    
    cout << "Initialized EdgeStream for folder: " << folder << " (base=" << base << ")" << endl;
    
    DIR* dir = opendir(folder.c_str());
    if (dir == nullptr) {
        cout << "Error: Cannot open directory '" << folder << "'. "
             << "Reason: " << strerror(errno) << endl;
        exit(-1);
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        string filename = entry->d_name;
        
        if (entry->d_type != DT_REG) continue;
        
        if (filename.find("stream_") == 0 && filename.find(".mtx") != string::npos) {
            batch_names.push_back(filename);
        }
    }
    
    closedir(dir);
    
    sort(batch_names.begin(), batch_names.end());

    cout << "Found " << batch_names.size() << " stream updates files: " << endl;
    if (!batch_names.empty()) {
        for (const string& filename : batch_names) {
            cout << "  " << filename << endl;
        }
    }
}

bool EdgeStream::loadNextBatch() {
    // Reset current batch state
    this->batch_edges.clear();
    int next_batch_index = this->batch_index + 1;
    
    // Check if we have more batches
    if (next_batch_index >= batch_names.size()) {
        cout << "No more stream files available." << endl;
        return false;
    }
    
    string next_batch_name = batch_names[next_batch_index];
    string next_batch_path = this->operations_folder + "/" + next_batch_name;

    cout << "Next stream file: " << next_batch_name << endl;

    if (manual_selection) {
        cout << "Load next stream file? (yes/yes_always/no): ";
        string response;
        cin >> response;
        
        if (response == "yes" || response == "y" || response == "Y") {
            // Load this file only
        } else if (response == "yes_always" || response == "ya" || response == "YA") {
            // Load this file and disable manual selection for future
            manual_selection = false;
            cout << "Automatic loading enabled for remaining batches." << endl;
        } else {
            // User chose not to load
            cout << "Skipping batch " << next_batch_index << endl;
            return false;
        }
    }
    
    // Load the batch file
    cout << "Loading: " << next_batch_path << endl;
    
    // Determine operation type from filename
    if (next_batch_name.find("insert") != string::npos || next_batch_name.find("inc") != string::npos) {
        this->current_op = OperationType::INCREMENTAL;
    } else if (next_batch_name.find("delete") != string::npos || next_batch_name.find("dec") != string::npos) {
        this->current_op = OperationType::DECREMENTAL;
    }
    
    // Load edges from file into batch_edges vector
    loadBatchFromFile(next_batch_path, current_op);
    
    // Update state
    this->batch_index = next_batch_index;
    this->batch_filename = next_batch_name;
    this->batch_size = this->batch_edges.size();
    this->total_edges_processed += this->batch_size;
    
    cout << "Loaded " << this->batch_size << " edges (" 
         << (current_op == OperationType::INCREMENTAL ? "INSERT" : "DELETE") 
         << ")" << endl;
    
    return true;
}



void EdgeStream::loadBatchFromFile(const string& filepath, OperationType op) {

    bool weightFlag = this->autoDetectWeightExist(filepath);
    // Open and map file
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        cout << "Error: Cannot open batch file '" << filepath << "'. "
             << "Reason: " << strerror(errno) << endl;
        exit(-1);
    }
    
    size_t file_size = fsize(filepath.c_str());
    char* ss_head = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(ss_head != MAP_FAILED);
    madvise(ss_head, file_size, MADV_SEQUENTIAL);
    
    char* ss = ss_head;
    size_t curr = 0, next = 0;
    int checkwt = 0;  // Counter for weight field parsing (src, dest, weight = 3 fields)
    size_t edge_count = 0;

    // PASS 1 count total edges in file
    while (next < file_size) {
        if (weightFlag != 0) checkwt++;  // Track field position for weight parsing

        // Skip current field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        // curr = next;

        if (checkwt != 3) edge_count++;  // Count edges (ignore weight fields)
        if (checkwt == 3) checkwt = 0;   // Reset after processing weight
    }

    edge_count = edge_count >> 1;
    this->batch_edges.resize(edge_count);   // Create edge_count elements for direct indexing

    // PASS 2: Parse and arrange edges in batch layout
    vertex_t source, dest;
    size_t offset = 0;  // Current edge being processed
    curr = 0;
    next = 0;
    weight_t wtvalue;

    while (offset < edge_count) {
        // Parse source vertex (convert from 1-based to 0-based)
        char* sss = ss + curr;
        source = atoi(sss) - this->base; // Convert from 1-based to 0-based indexing

        // Jump to destination field
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        
        // Parse destination vertex (convert from 1-based to 0-based)
        char* sss1 = ss + curr;
        dest = atoi(sss1) - this->base; // Convert from 1-based to 0-based indexing

        // Skip to weight field (if present)
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        
        // Parse weight if present
        if (weightFlag != 0) {
            curr = next;
            char* sss1 = ss + curr;
            wtvalue = atof(sss1);

            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }
        else{
            wtvalue = 1.0;  // Default weight
        }
        curr = next;

        batch_edges[offset] = make_tuple(source, dest, wtvalue);  // Store edge tuple

        offset++;
    }

    // Clean up memory mapping
    munmap(ss_head, file_size);
    close(fd);
}

bool EdgeStream::autoDetectWeightExist(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Cannot open file '" << filename << "' for weight detection." << endl;
        exit(-1);
    }
    
    string line;
    if (!getline(file, line)) {
        cout << "Error: Cannot read first line from file '" << filename << "'." << endl;
        file.close();
        exit(-1);
    }
    
    file.close();
    
    // Check for comment line
    if (!line.empty() && line[0] == '%') {
        cout << "Error: Comment lines not supported in batch files. Found: " << line << endl;
        exit(-1);
    }
    
    // Try to parse elements
    double num1, num2, num3;
    int parsed = sscanf(line.c_str(), "%lf %lf %lf", &num1, &num2, &num3);
    
    if (parsed == 3) {
        return true;  // Edge weights exist
    } else if (parsed == 2) {
        return false; // No edge weights
    } else {
        cout << "Error: Invalid line format. Expected 2 or 3 numbers, got " << parsed << " numbers." << endl;
        cout << "First line content: '" << line << "'" << endl;
        exit(-1);
    }
}


EdgeStream::~EdgeStream() {

    cout << "EdgeStream cleanup complete." << endl;
}

 
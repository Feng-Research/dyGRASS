
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

class GPU_Stream_Edges{
    private:
        size_t max_capacity;
    public:
        size_t batch_size;
        vertex_t * edges;
        weight_t * weights;

        vertex_t *edges_device;
        weight_t *weights_device;

        

        GPU_Stream_Edges(size_t max_capacity){
            this->max_capacity = max_capacity;
            this->batch_size = 0;
            this->edges = new vertex_t[max_capacity * 2];
            this->weights = new weight_t[max_capacity];

            HRR(cudaMalloc((void **)&edges_device, sizeof(vertex_t)*max_capacity * 2));
            HRR(cudaMalloc((void **)&weights_device, sizeof(weight_t)*max_capacity));
        }

        void ensureCapacity(){

        }

        ~GPU_Stream_Edges(){
            delete[] edges;
            delete[] weights;

            cudaFree(edges_device);
            cudaFree(weights_device);
            cout << "GPU_Stream_Edges deallocated" << endl;
        }

};

class GPU_Dual_Graph{

    public:
        // === Shared Properties ===
        vertex_t vertex_num;                    // Number of vertices in both graphs
        long multiplier;                        // Hash key multiplier for edge mapping
        
        //incremental
        float distortion;
        //decremental

        // === Sparse Graph Properties (Target Sparsified Graph) ===
        index_t sparse_edge_num;                // Current number of edges in sparse graph
        unordered_map<long, pair<index_t,index_t>> * sparse_map; // Edge mapping for O(1) operations
        
        // Sparse Graph CSR Data (Host)
        vertex_t * sparse_degree_list;          // Current degrees (updated during edge removal)
        vertex_t * sparse_degree_original;      // Original degrees (for memory layout)
        weight_t ** sparse_beg_ptr;             // Pointers to neighbor data blocks
        weight_t ** sparse_beg_ptr_device_content; // Device addresses for GPU pointers
        weight_t * sparse_neighbors_data;       // Packed neighbor data (adj and weights/resistance)

        
        // Sparse Graph GPU Memory
        vertex_t * sparse_degree_list_device;   // GPU copy of current degrees
        vertex_t * sparse_degree_original_device; // GPU copy of original degrees
        weight_t ** sparse_beg_ptr_device;      // GPU pointer array
        weight_t * sparse_neighbors_data_device; // GPU neighbor data
        unsigned  sparse_extra_neighbor_offset; // Offset in extra space 

        // === Dense Graph Properties (Updated Dense Graph for Pathfinding) ===
        index_t dense_edge_num;                 // Current number of edges in dense graph
        unordered_map<long, pair<index_t,index_t>> * dense_map; // Edge mapping for O(1) operations
        
        // Dense Graph CSR Data (Host)
        vertex_t * dense_degree_list;           // Current degrees (updated during edge removal)
        vertex_t * dense_degree_original;       // Original degrees (for memory layout)
        weight_t ** dense_beg_ptr;              // Pointers to neighbor data blocks
        weight_t ** dense_beg_ptr_device_content; // Device addresses for GPU pointers
        weight_t * dense_neighbors_data;        // Packed neighbor data
        
        // Dense Graph GPU Memory
        vertex_t * dense_degree_list_device;    // GPU copy of current degrees
        vertex_t * dense_degree_original_device; // GPU copy of original degrees
        weight_t ** dense_beg_ptr_device;       // GPU pointer array
        weight_t * dense_neighbors_data_device; // GPU neighbor data
        unsigned  dense_extra_neighbor_offset;  // Offset in extra space


        GPU_Dual_Graph(
            CSRGraph& dense_ginst, CSRGraph& sparse_ginst
        ){  
            // Note: Current implementation optimized for decremental processing
            // shared properties
            this->vertex_num = dense_ginst.vert_count;
            assert(dense_ginst.vert_count == sparse_ginst.vert_count);

            this->multiplier = sparse_ginst.multiplier;
            assert(sparse_ginst.multiplier == dense_ginst.multiplier);
            this->sparse_map = &sparse_ginst.edge_map; // don't deallocate this



            // dense graph properties
            this->dense_edge_num = dense_ginst.edge_count;
            this->dense_map = &dense_ginst.edge_map; // don't deallocate this
            this->dense_degree_list = dense_ginst.degree_list; // don't deallocate this
            this->dense_degree_original = new vertex_t[vertex_num];
            for (int i = 0; i < vertex_num; i++){
                dense_degree_original[i] = dense_ginst.degree[i];
            }
            auto[beg_ptr, neighbors_data] = createNeighborArray(dense_ginst);
            this->dense_beg_ptr = beg_ptr;
            this->dense_beg_ptr_device_content = new weight_t*[vertex_num]; // store device pointers
            this->dense_neighbors_data = neighbors_data;

            HRR(cudaMalloc((void **)&dense_degree_list_device, sizeof(vertex_t)*vertex_num));
            HRR(cudaMalloc((void **)&dense_degree_original_device, sizeof(vertex_t)*vertex_num));
            HRR(cudaMalloc((void ***)&dense_beg_ptr_device, sizeof(weight_t*)*vertex_num));
            HRR(cudaMalloc((void **)&dense_neighbors_data_device, sizeof(weight_t)*dense_edge_num*2 * 2));
            // dense_extra_neighbors_data_device = dense_neighbors_data_device + dense_edge_num*4;
            int offset = 0;
            for (int i = 0; i < vertex_num; i++){
                // weight_t * ptr = dense_neighbors_data_device + offset;
                // dense_beg_ptr_device_content[i] = reinterpret_cast<weight_t*>(ptr);
                dense_beg_ptr_device_content[i] = dense_neighbors_data_device + offset;
                offset += dense_degree_original[i]*2;
            }


        }
        

        tuple<weight_t**, weight_t*> createNeighborArray(const CSRGraph& graph) {
            // Allocate an array of pointers, one for each vertex
            weight_t** beg_ptr = new weight_t* [graph.vert_count];

            weight_t* neighbors_data = new weight_t[graph.edge_count * 2 * 2]; // Allocate space for both indices, weights and from
            size_t offset = 0;
            for (size_t i = 0; i < graph.vert_count; i++) {
                // Set the pointer to the beginning of the adjacency list for the vertex
                beg_ptr[i] = neighbors_data + offset;
                // Copy the adjacency list for the vertex
                index_t degree = graph.degree[i];
                for (size_t j = graph.begin[i]; j < graph.begin[i + 1]; j++) {
                    neighbors_data[offset] = static_cast<weight_t>(graph.adj[j]);
                    neighbors_data[offset + degree] = graph.weight[j];
                    offset++;
                }
                offset += degree;

            }

            return {beg_ptr, neighbors_data};

        }

        void IncrementalUpdateFilteredEdges(
            int* path_selected_flag, 
            float* final_conductance,
            int edge
        ){
            // Process results for current batch of samples
            for (int i = 0; i < sample_count; i++){
                int find_at = path_selected_flag[i]; // -1 if no path found
                weight_t weight = sample_weights[sample_ptr/2 + i];
                
                // === Path Found: Edge is Successfully Sparsified ===
                if(find_at != -1){ 
                    // Do nothing - the sample edge is represented by the found path
                    // This is the core sparsification: avoid adding redundant edges
                }
                // === No Path Found: Must Add Edge to Maintain Connectivity ===
                else{ 
                    no_path_count += 1;

                    // Get source and destination from sample array layout
                    int source = sample_nodes_all[sample_ptr + i];
                    int target = sample_nodes_all[sample_ptr + sample_count + i];
                    
                    // Add edge to extension array in triplet format
                    ext[ext_size*3] = source;
                    ext[ext_size*3 + 1] = target;
                    ext[ext_size*3 + 2] = weight;

                    // Update vertex degrees (increment before use)
                    int degree_source = ++degree_list[source];
                    int degree_target = ++degree_list[target];

                    // Add bidirectional edges to neighbor arrays
                    manageExtraNeighborsData(source, target, weight, degree_source, degree_target);
                    manageExtraNeighborsData(target, source, weight, degree_target, degree_source);

                    ext_size += 1;
                }
            }
            cout << "no path count: " << no_path_count << endl;

            // === Sync Updated Graph Back to GPU ===
            HRR(cudaMemcpy(neighbors_data_device, neighbors_data, sizeof(weight_t)*edge_count*4 * 2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(beg_ptr_device, beg_ptr_device_content, sizeof(weight_t*)*vert_count, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(degree_list_device, degree_list, sizeof(vertex_t)*vert_count, cudaMemcpyHostToDevice));
        }

        bool removeAndUpdateDelEdges(){

            int count = 0;
            long key, a, b;

            // adjust sample_num for the last batch
            if (this->decremental_sample_ptr + 2 * this->decremental_sample_num >= this->decremental_edge_num * 2){
                this->decremental_sample_num = this->decremental_edge_num - this->decremental_sample_ptr / 2;
            }
           
            for(int i = 0; i < this->decremental_sample_num; i++){

                a = decremental_edges[decremental_sample_ptr + i];
                b = decremental_edges[decremental_sample_ptr + decremental_sample_num + i];
                if(a > b) swap(a, b);
                key = a * this->multiplier + b;

                if(this->dense_map->count(key) ==0){
                    cout << "dense_map not found: a: " << a << " b: " << b << " key: " << key << endl;
                }
                if(this->dense_map->count(key) > 0){ 
                    // if this edge is in the dense graph, remove it from dense

                    pair<vertex_t,vertex_t> value = this->dense_map->at(key);
                    index_t index_b_a = value.first; // b at a
                    index_t index_a_b = value.second; // a at b

                    dense_map->erase(key);
                    // dense graph: 
                    // beg_ptr don't need to be updated, because the degree shrunk
                    // degree_list need to be updated
                    int degree_a = this->dense_degree_list[a] -= 1;
                    int degree_b = this->dense_degree_list[b] -= 1;
                    // neighbors_data need to be updated. //col, weight, from, reverse
                    assert(degree_a != 0);
                    assert(degree_b != 0);
                    int interval_a = this->dense_degree_original[a];
                    int interval_b = this->dense_degree_original[b];

                    // if it's last edge in the vertex, don't need to update the neighbors_data
                    if(degree_a != index_b_a){
                        // else put the deleted edge to the last position, ignore it by --degree
                        vertex_t a_last = this->dense_beg_ptr[a][degree_a];
                        this->dense_beg_ptr[a][index_b_a] = a_last; // col
                        this->dense_beg_ptr[a][index_b_a + interval_a] = this->dense_beg_ptr[a][degree_a + interval_a]; // weight


                        if(a<a_last){
                            long key_last = a * this->multiplier + a_last;
                            this->dense_map->at(key_last).first = index_b_a;
                        }
                        else{
                            long key_last = a_last * this->multiplier + a;
                            this->dense_map->at(key_last).second = index_b_a;
                        }

                    }

                    if(degree_b != index_a_b){
                        vertex_t b_last = this->dense_beg_ptr[b][degree_b];
                        this->dense_beg_ptr[b][index_a_b] = b_last; // col
                        this->dense_beg_ptr[b][index_a_b + interval_b] = this->dense_beg_ptr[b][degree_b + interval_b]; // weight

                        if(b < b_last){
                            long key_last = b * this->multiplier + b_last;
                            this->dense_map->at(key_last).first = index_a_b;
                        }
                        else{
                            long key_last = b_last * this->multiplier + b;
                            this->dense_map->at(key_last).second = index_a_b;
                        }
                    }

                   

                    if(this->sparse_map->count(key) ==0){
                     cout << "sparse_map not found: a: " << a << " b: " << b << " key: " << key << endl;
                    }
                    // sparse graph: mtx_sparse
                    // if this edge is in the sparse graph, remove it from sparse (not necessarily in the sparse graph)
                    if (this->sparse_map->count(key) > 0){
                        
                        // pair<vertex_t,vertex_t> value = this->sparse_map->at(key);
                        // index_t index_b_a = value.first; // b at a
                        // index_t index_a_b = value.second; // a at b
                        this->sparse_map->erase(key);

                        // degree_a = this->sparse_degree_list[a] -= 1;
                        // degree_b = this->sparse_degree_list[b] -= 1;
                        // assert(degree_a != 0);
                        // assert(degree_b != 0);
                        // interval_a = this->sparse_degree_original[a];
                        // interval_b = this->sparse_degree_original[b];

                        // if(degree_a != index_b_a){
                        //     vertex_t a_last = this->sparse_beg_ptr[a][degree_a];
                        //     this->sparse_beg_ptr[a][index_b_a] = a_last; // col
                        //     this->sparse_beg_ptr[a][index_b_a + interval_a] = this->sparse_beg_ptr[a][degree_a + interval_a]; // weight

                        //     if(a < a_last){
                        //         long key_last = a * this->multiplier + a_last;
                        //         this->sparse_map->at(key_last).first = index_b_a;
                        //     }
                        //     else{
                        //         long key_last = a_last * this->multiplier + a;
                        //         this->sparse_map->at(key_last).second = index_b_a;
                        //     }
                        // }

                        // if(degree_b != index_a_b){
                        //     vertex_t b_last = this->sparse_beg_ptr[b][degree_b];
                        //     this->sparse_beg_ptr[b][index_a_b] = b_last; // col
                        //     this->sparse_beg_ptr[b][index_a_b + interval_b] = this->sparse_beg_ptr[b][degree_b + interval_b]; // weight

                        //     if(b < b_last){
                        //         long key_last = b * this->multiplier + b_last;
                        //         this->sparse_map->at(key_last).first = index_a_b;
                        //     }
                        //     else{
                        //         long key_last = b_last * this->multiplier + b;
                        //         this->sparse_map->at(key_last).second = index_a_b;
                        //     }
                        // }

                        // only add the edge to filted_sample_nodes if it was in the sparse graph, then it will be used for random walk
                        filted_del_sample_edges[count] = a;
                        filted_del_sample_edges[count + decremental_sample_num ] = b;
                        count += 1;
                    } // if exist in sparse graph end
                } // if exist in dense graph end
            } // for loop end

            this->filtered_del_sample_count = count; // read from outside to determine the block num
            // update the sample_ptr, copy the filted_del_sample_edges to device
            if (this->decremental_sample_ptr != 0){ // first time, copy graph handle by other function
                    HRR(cudaMemcpy(dense_degree_list_device, dense_degree_list, sizeof(vertex_t)*vertex_num, cudaMemcpyHostToDevice));
                    HRR(cudaMemcpy(dense_neighbors_data_device, dense_neighbors_data, sizeof(weight_t)*dense_edge_num*4 * 2, cudaMemcpyHostToDevice));
            }
            HRR(cudaMemcpy(decremental_sample_nodes_device, filted_del_sample_edges, sizeof(vertex_t)*decremental_sample_num*2, cudaMemcpyHostToDevice));
            
            this->decremental_sample_ptr += this->decremental_sample_num * 2;

            if (this->decremental_sample_ptr < this->decremental_edge_num * 2){ 
                return false;
            }
            else{
                return true;
            }
            
        } 
        
        void copyDualGraphToDevice(){
            HRR(cudaMemcpy(sparse_degree_list_device, sparse_degree_list, sizeof(vertex_t)*vertex_num, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sparse_degree_original_device, sparse_degree_original, sizeof(vertex_t)*vertex_num, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sparse_beg_ptr_device, sparse_beg_ptr_device_content, sizeof(weight_t*)*vertex_num, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sparse_neighbors_data_device, sparse_neighbors_data, sizeof(weight_t)*sparse_edge_num*2 * 2, cudaMemcpyHostToDevice));

            HRR(cudaMemcpy(dense_degree_list_device, dense_degree_list, sizeof(vertex_t)*vertex_num, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(dense_degree_original_device, dense_degree_original, sizeof(vertex_t)*vertex_num, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(dense_beg_ptr_device, dense_beg_ptr_device_content, sizeof(weight_t*)*vertex_num, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(dense_neighbors_data_device, dense_neighbors_data, sizeof(weight_t)*dense_edge_num*2 * 2, cudaMemcpyHostToDevice));
        }

        /**
         * Process random walk results and update sparse graph incrementally
         * 
         * After random walks complete, this method processes the results:
         * 
         * 1. **Path Found**: Reconstruct and add alternative path edges to sparse graph
         * 2. **No Path**: Mark edge for heuristic recovery (essential for connectivity)
         * 3. **Edge Recovery**: Add intermediate path edges to maintain connectivity
         * 4. **Tracking**: Record all added edges for analysis and output
         * 
         * Key Logic: If random walk finds alternative path, the removed edge can be
         * safely omitted from sparse graph since connectivity is preserved via the path.
         * 
         * @param path_selected Array of path vertices for each successful walk  
         * @param path_selected_flag Array indicating if path was found (-1 if not)
         * @param final_conductance Array of path conductance values
         * @param n_steps Maximum steps per path
         */
        void incrementalUpdateSparse(int* path_selected, int* path_selected_flag, float* final_conductance, int n_steps){
            int interval = n_steps;
            int count = 0;
            long a, b, key;
            int add_count = 0;
            for (int i = 0; i < this->filtered_del_sample_count; i++){

                int find_at = path_selected_flag[i]; // include source and target

                int source = this->filted_del_sample_edges[i];
                int target = this->filted_del_sample_edges[i + decremental_sample_num];
                int from = path_selected[interval*i];
                
                if (find_at != -1){
                    // recover these edges to sparse_map
                    assert(from == source);
                    assert(path_selected[interval*i + find_at] == target);
                    for (int j = 0; j < find_at; j++){

                        int to = path_selected[interval*i + j + 1];
                        
                        a = from;
                        b = to;
                        if(a > b) swap(a, b);
                        key = a * this->multiplier + b;
                        if (sparse_map->count(key) == 0){
                            add_count += 1;
                            sparse_map->insert({key, {0,0}});
                            added_edges->insert(key);
                        }

                        from = to;
                    }
                    
                }
                else{
                    // mark these edges for heuristic recovery
                    // cout << "No path for edge: " << source << " " << target << endl;
                    this->decremental_no_path_count += 1;
                    heuristic_sample_nodes[count] = source;
                    heuristic_sample_nodes[count + 1] = target;
                    count += 2;
                }
            }
            cout << "No path count: " << count/2 << endl;
            cout << "Added edges: " << add_count << endl;
            this->heuristic_sample_num = count;
            if(count > 0){
                HRR(cudaMemcpy(heuristic_sample_nodes_device, heuristic_sample_nodes, sizeof(vertex_t)*heuristic_sample_num, cudaMemcpyHostToDevice));
            }
        }

        /**
         * Heuristic Recovery: Add essential edges using short random walks
         * 
         * For edges where no alternative paths were found during main processing,
         * this method performs short random walks to find minimal connectivity paths:
         * 
         * 1. **Short Walks**: Perform multiple short (3-step) walks from each endpoint
         * 2. **Path Addition**: Add all intermediate edges from successful walks
         * 3. **Connectivity Insurance**: Ensures graph remains connected despite aggressive removal
         * 
         * This is a fallback mechanism to preserve essential connectivity when
         * the main decremental algorithm cannot find alternative paths.
         * 
         * @param path_selected Array containing short walk paths
         * @param n_walker Number of parallel walkers per endpoint
         */
        void heuristicRecovery(int * path_selected, int n_walker){
            int steps = 3;
            int num = this->heuristic_sample_num;
            // for each edge in the filted_sample_nodes, if it is not in the sparse_map, add it to the sparse_map

            for (int i = 0; i < num; i ++){
                int source = heuristic_sample_nodes[i];
                int start_at = i * steps * n_walker;
                // int from = path_selected[start_at];
                // assert(from == source);
                for (int j = 0; j < n_walker; j++){

                    int from = path_selected[start_at + j * steps];
                    assert(from == source);

                    for (int k = 1; k < steps; k++){

                        int to = path_selected[start_at + j * steps + k];
                        int a = from;
                        int b = to;
                        if(a > b) swap(a, b);
                        long key = a * this->multiplier + b;
                        if (sparse_map->count(key) == 0){
                            sparse_map->insert({key, {0,0}});
                        }
                        from = to;
                    }
                }
            }
            // for (int i = 0; i < this->filted_sample_count; i++){
            //     int a = filted_sample_nodes[i];
            //     int b = filted_sample_nodes[i + sample_count];
            //     if(a > b) swap(a, b);
            //     long key = a * this->multiplier + b;
            //     if (sparse_map->count(key) == 0){
            //         sparse_map->insert({key, 0});
            //     }
            // }
        }

        void sparse_map_to_1_based_mtx(string file_name){
            std::ofstream txt_file(file_name, std::ios::out);
            if(txt_file.is_open()){
                for(auto it = sparse_map->begin(); it != sparse_map->end(); ++it){

                    auto key = it->first;
                    long a = key / this->multiplier;
                    long b = key % this->multiplier;
                    txt_file << a + 1 << " " << b + 1 << " " << 1 << std::endl;
                }
            }
            cout<< "Updated sparse mtx file saved to: " << file_name << endl; 
                
        }

        // saved edges are 1-based
        void save_added_edges(string file_name){
            cout << "Size of added edges: " << added_edges->size() << endl;
            std::ofstream txt_file(file_name, std::ios::out);
            if(txt_file.is_open()){
                for(auto it = added_edges->begin(); it != added_edges->end(); ++it){

                    long key = *it;
                    long a = key / this->multiplier;
                    long b = key % this->multiplier;
                    txt_file << a + 1 << " " << b + 1 << std::endl;
                }
            }
            cout<< "Added edges saved to: " << file_name << endl; 
        }

        ~GPU_Dual_Graph(){ 
            // shared properties
            delete[] heuristic_sample_nodes;
            cudaFree(heuristic_sample_nodes_device);
            delete added_edges;

            // del edge
            delete[] filted_del_sample_edges;
            cudaFree(decremental_sample_nodes_device);

            // sparse graph
            // delete[] sparse_array_mtx;
            // delete[] sparse_array_ext;
            // delete[] sparse_degree_original;
            // delete[] sparse_beg_ptr;
            // delete[] sparse_neighbors_data;
            // delete[] sparse_beg_ptr_device_content;
            // cudaFree(sparse_degree_list_device);
            // cudaFree(sparse_degree_original_device);
            // cudaFree(sparse_beg_ptr_device);
            // cudaFree(sparse_neighbors_data_device);

            // dense graph  
            delete[] dense_degree_original;
            delete[] dense_beg_ptr;
            delete[] dense_neighbors_data;
            delete[] dense_beg_ptr_device_content;
            cudaFree(dense_degree_list_device);
            cudaFree(dense_degree_original_device);
            cudaFree(dense_beg_ptr_device);
            cudaFree(dense_neighbors_data_device);
        }
        
};



#endif

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


# define DENSE 0
# define SPARSE 1

class GPU_Stream_Edges{
    private:
        size_t max_capacity;
        bool overflow_flag;
        size_t  next_run_index;
       

    public:
        int n_steps;
        size_t batch_size;
        size_t load_size;
        vertex_t * edges;
        weight_t * weights;
        int * path_selected; // selected path
        int * path_selected_flag; // path exist?


        // Device memory
        vertex_t *edges_device;
        weight_t *weights_device;
        int *path_selected_device;
        int *path_selected_flag_device;


        //heursistic recovery
        index_t heuristic_sample_num;
        vertex_t * heuristic_sample_nodes;      // Host: edges that need heuristic paths
        vertex_t * heuristic_sample_nodes_device; // GPU: edges for heuristic processing

        //reduction calculation
        //TODO: try reduction without global memory

        //Others
        OperationType current_op;

        

        GPU_Stream_Edges(size_t max_capacity, int n_steps){
            this->n_steps = n_steps;
            this->max_capacity = max_capacity;
            this->overflow_flag = false;
            this->next_run_index = 0;
            this->load_size = 0;
            this->batch_size = 0;
            this->current_op = INCREMENTAL;
            this->edges = nullptr;
            this->weights = nullptr;
            this->path_selected = new int[max_capacity * n_steps];
            this->path_selected_flag = new int[max_capacity];

            HRR(cudaMalloc((void **)&edges_device, sizeof(vertex_t)*max_capacity * 2));
            HRR(cudaMalloc((void **)&weights_device, sizeof(weight_t)*max_capacity));

            HRR(cudaMalloc((void **)&path_selected_device, sizeof(int)*max_capacity * n_steps));
            HRR(cudaMalloc((void **)&path_selected_flag_device, sizeof(int)*max_capacity));

        }

        void loadEdgeFromStream(const EdgeStream& edge_stream){

            this->batch_size = edge_stream.batch_size;
            assert(batch_size == edge_stream.batch_edges.size());
            this->current_op = edge_stream.current_op;

            if (this->edges != nullptr && this->weights != nullptr){
                delete[] this->edges;
                delete[] this->weights;
            }

            this->edges = new vertex_t[batch_size * 2];
            this->weights = new weight_t[batch_size];

            for (size_t i = 0; i < batch_size; i++){
                auto [src, dst, weight] = edge_stream.batch_edges[i];


                this->edges[i * 2] = src;
                this->edges[i * 2 + 1] = dst;
                this->weights[i] = weight;
            }

        }

        void loadEdgesToDevice(){
            if (this->edges == nullptr || this->weights == nullptr){
                cout << "Error: edges or weights not loaded." << endl;
                std::cerr << "Edges not loaded to GPU stream." << std::endl;
                exit(EXIT_FAILURE);
                return;
            }

            if (this->batch_size - this->next_run_index > this->max_capacity){

                this->overflow_flag = true;
                this->load_size = this->max_capacity;
                this->next_run_index += this->max_capacity;
                
            }
            else{

                this->overflow_flag = false;
                this->load_size = this->batch_size - this->next_run_index;  
                this->next_run_index = 0; // reset for next batch
            }

            HRR(cudaMemcpy(edges_device, edges + (next_run_index * 2), sizeof(vertex_t) * load_size * 2, cudaMemcpyHostToDevice));
            if (this->current_op == INCREMENTAL){
                HRR(cudaMemcpy(weights_device, weights + next_run_index, sizeof(weight_t) * load_size, cudaMemcpyHostToDevice));
            }

        }

        void downloadResult(){
            if (this->current_op == DECREMENTAL){
                HRR(cudaMemcpy(path_selected, path_selected_device, sizeof(int) * load_size * n_steps, cudaMemcpyDeviceToHost));
            }
            HRR(cudaMemcpy(path_selected_flag, path_selected_flag_device, sizeof(int) * load_size, cudaMemcpyDeviceToHost));
        }


        ~GPU_Stream_Edges(){
            delete[] edges;
            delete[] weights;
            delete[] path_selected;
            delete[] path_selected_flag;

            cudaFree(edges_device);
            cudaFree(weights_device);
            cudaFree(path_selected_device);
            cudaFree(path_selected_flag_device);
            cout << "GPU_Stream_Edges deallocated" << endl;
        }

};

class GPU_Dual_Graph{

    public:
        // === Shared Properties ===
        CSRGraph * graphs[2];
        vertex_t vertex_num;                    // Number of vertices in both graphs
        long multiplier;                        // Hash key multiplier for edge mapping
        
        //incremental
        //decremental
        index_t * wait_add_edges;

        // try new thing
        index_t edge_num[2];
        unordered_map<long, pair<index_t,index_t>> * graph_map[2]; // Edge mapping for O(1) operations
        vertex_t * degree_list[2];          // Current degrees (updated during edge removal)
        vertex_t * bin_size[2];              // Original degrees (for memory layout)
        weight_t ** beg_ptr[2];             // Pointers to neighbor data blocks
        weight_t ** beg_ptr_device_content[2]; // Device addresses for GPU pointers
        weight_t * concatenated_neighbors_data[2]; // Concatenated neighbor data
        weight_t * extra_neighbor_data[2];       // Extra neighbor data for dynamic updates

        vertex_t * degree_list_device[2];   // GPU copy of current degrees
        vertex_t * bin_size_device[2];       // GPU copy of original degrees
        weight_t ** beg_ptr_device[2];       // GPU pointer array
        weight_t * concatenated_neighbors_data_device[2]; // GPU neighbor data
        weight_t * extra_neighbor_data_device[2];       // GPU extra neighbor data

        unsigned extra_neighbor_offset[2];


        GPU_Dual_Graph(
            CSRGraph* dense_ginst, CSRGraph* sparse_ginst
        ){  
            this->graphs[DENSE] = dense_ginst;
            this->graphs[SPARSE] = sparse_ginst;

            assert(graphs[DENSE]->vert_count == graphs[SPARSE]->vert_count);
            this->vertex_num = graphs[DENSE]->vert_count;

            assert(graphs[DENSE]->multiplier == graphs[SPARSE]->multiplier);
            this->multiplier = graphs[SPARSE]->multiplier;

            for (int graph_type = 0; graph_type < 2 ; graph_type++){
                
                this->edge_num[graph_type] = graphs[graph_type]->edge_count;
                this->graph_map[graph_type] = &graphs[graph_type]->edge_map; // don't deallocate this
                this->degree_list[graph_type] = graphs[graph_type]->degree_list; // don't deallocate this
                this->bin_size[graph_type] = new vertex_t[vertex_num];

                for (int i = 0; i < vertex_num; i++){
                    this->bin_size[graph_type][i] = graphs[graph_type]->degree_list[i];
                }
                auto[beg_ptr, neighbors_data] = createNeighborArray(graphs[graph_type]);
                this->beg_ptr[graph_type] = beg_ptr;
                this->concatenated_neighbors_data[graph_type] = neighbors_data;
                this->beg_ptr_device_content[graph_type] = new weight_t*[vertex_num]; // store device pointers

                HRR(cudaMalloc((void **)&degree_list_device[graph_type], sizeof(vertex_t)*vertex_num));
                HRR(cudaMalloc((void **)&bin_size_device[graph_type], sizeof(vertex_t)*vertex_num));
                HRR(cudaMalloc((void ***)&beg_ptr_device[graph_type], sizeof(weight_t*)*vertex_num));
                HRR(cudaMalloc((void **)&concatenated_neighbors_data_device[graph_type], sizeof(weight_t)*edge_num[graph_type]*2 * 2));
                int offset = 0;
                for (int i = 0; i < vertex_num; i++){
                    beg_ptr_device_content[graph_type][i] = concatenated_neighbors_data_device[graph_type] + offset;
                    offset += degree_list[graph_type][i]*2;
                }

                this->extra_neighbor_data[graph_type] = this->concatenated_neighbors_data[graph_type] + this->edge_num[graph_type]*2;
                this->extra_neighbor_data_device[graph_type] = this->concatenated_neighbors_data_device[graph_type] + offset;
                assert(extra_neighbor_data_device == concatenated_neighbors_data_device + this->edge_num[graph_type]*2);
                this->extra_neighbor_offset[graph_type] = 0;

                
            }
            // copy to GPU
            this->updateDeviceDualGraph();

        }
        

        tuple<weight_t**, weight_t*> createNeighborArray(const CSRGraph * graph) {
            // Allocate an array of pointers, one for each vertex
            weight_t** beg_ptr = new weight_t* [this->vertex_num];

            weight_t* neighbors_data = new weight_t[graph->edge_count * 2 * 2]; // Allocate space for both indices, weights and from
            size_t offset = 0;
            for (size_t i = 0; i < graph->vert_count; i++) {
                // Set the pointer to the beginning of the adjacency list for the vertex
                beg_ptr[i] = neighbors_data + offset;
                // Copy the adjacency list for the vertex
                index_t degree = graph->degree[i];
                for (size_t j = graph->begin[i]; j < graph->begin[i + 1]; j++) {
                    neighbors_data[offset] = static_cast<weight_t>(graph->adj[j]);
                    neighbors_data[offset + degree] = graph->weight[j];
                    offset++;
                }
                offset += degree;

            }

            return {beg_ptr, neighbors_data};

        }


        void preprocessStreamEdges(EdgeStream & stream_edges){

            OperationType op = stream_edges.current_op;

            auto it = stream_edges.batch_edges.begin();
            while(it != stream_edges.batch_edges.end()){

                auto [u, v, w] = *it;
                if (u > v) swap(u, v);
                long key = u * this->multiplier + v;

                if (op == INCREMENTAL) {

                   if (this->graph_map[DENSE]->count(key) > 0) {
                        // edge already exist so add weights to exist edges
                        cout << "Warning: Incremental Edge already exists in original graph, skip: " << u << " " << v << endl;
                        it = stream_edges.batch_edges.erase(it);

                        // do edge weights update here, in the function automatically check sparsifier  has this edge or not
                        this->dualGraphEdgeweightsUpdate(u, v, w);

                    } else {
                        // edge not exist in original graph as expected

                        if (this->graph_map[SPARSE]->count(key) > 0) {
                            // illegal state
                            cerr << "Error: Edge not exists in original graph, but in sparsifier: " << u << " " << v << endl;
                            exit(EXIT_FAILURE);
                        }

                        this->edgeInsertion(u, v, w, DENSE);

                        ++it; // keep it for NBRW
                    }

                } else if (op == DECREMENTAL) {

                    if (this->graph_map[DENSE]->count(key) == 0) {

                        // edge not exist in original graph, just skip it
                        cout << "Warning: Decremental Edge not exist in original graph, skip: " << u << " " << v << endl;
                        it = stream_edges.batch_edges.erase(it);

                        if (this->graph_map[SPARSE]->count(key) > 0) {
                            // illegal state
                            cerr << "Error: Edge not exists in original graph, but exists in sparsifier: " << u << " " << v << endl;
                            exit(EXIT_FAILURE);
                        }

                    } else {
                        // edge exists in original graph, as expected
                        // do edge deletion for dense graph here
                        this->edgeDeletion(u, v, DENSE);
                        if (this->graph_map[SPARSE]->count(key) > 0) {
                            // edge also exists in sparsifier
                            // do edge deletion for sparse graphs here
                            this->edgeDeletion(u, v, SPARSE);
                            ++it; // keep it for NBRW
                            
                        } else {
                            // edge not exist in sparsifer, just remove it from dense, not need NBRW
                            it = stream_edges.batch_edges.erase(it);
                        }
                        
                    }

                }
            }
            // update to GPU
            this->updateDeviceDualGraph();
        }

        void updateDeviceDualGraph(){
            for (int graph_type = 0; graph_type < 2; graph_type++){
                HRR(cudaMemcpy(degree_list_device[graph_type], degree_list[graph_type], sizeof(vertex_t)*vertex_num, cudaMemcpyHostToDevice));
                HRR(cudaMemcpy(bin_size_device[graph_type], bin_size[graph_type], sizeof(vertex_t)*vertex_num, cudaMemcpyHostToDevice));
                HRR(cudaMemcpy(beg_ptr_device[graph_type], beg_ptr_device_content[graph_type], sizeof(weight_t*)*vertex_num, cudaMemcpyHostToDevice));
                HRR(cudaMemcpy(concatenated_neighbors_data_device[graph_type], concatenated_neighbors_data[graph_type], sizeof(weight_t)*edge_num[graph_type]*2 * 2, cudaMemcpyHostToDevice));
            }
        }


        void dualGraphEdgeweightsUpdate(vertex_t a, vertex_t b, weight_t added_weight){
            long key = a * this->multiplier + b;
            index_t b_index_in_a, a_index_in_b;

            for (int graph_type = 0; graph_type < 2; graph_type++) {
                if (this->graph_map[graph_type]->count(key) > 0) {
                    b_index_in_a = this->graph_map[graph_type]->at(key).first;
                    a_index_in_b = this->graph_map[graph_type]->at(key).second;

                    this->beg_ptr[graph_type][a][b_index_in_a] += added_weight;
                    this->beg_ptr[graph_type][b][a_index_in_b] += added_weight;
                }
            }
        }

        void edgeInsertion(vertex_t a, vertex_t b, weight_t weight, int graph_type){

            if (a > b) swap(a, b);
            long key = a * this->multiplier + b;
            int degree_a, degree_b;

            if (this->graph_map[graph_type]->count(key) == 0){

                degree_a = ++this->degree_list[graph_type][a];
                degree_b = ++this->degree_list[graph_type][b];


                this->graph_map[graph_type]->emplace(key, make_pair(degree_a -1, degree_b -1));

                edgeInsertionForNeighborData(a, b, degree_a, weight, graph_type);
                edgeInsertionForNeighborData(b, a, degree_b, weight, graph_type);

            }else{
                //error
                cerr << "Error: Edge already exists in graph map: " << a << " " << b << endl;
                exit(EXIT_FAILURE);
            }

        }


        void edgeInsertionForNeighborData(vertex_t a, vertex_t b, int degree_a, weight_t weight, int graph_type){

            int bin_a = this->bin_size[graph_type][a];
    
            if (degree_a <= bin_a) {
                this->beg_ptr[graph_type][a][degree_a - 1] = b;
                this->beg_ptr[graph_type][a][degree_a - 1 + bin_a] = weight;
            } else {
                assert(degree_a == bin_a + 1);
                this->bin_size[graph_type][a] += 1; // expand bin size

                weight_t *ptr_old = this->beg_ptr[graph_type][a];
                weight_t *ptr_new = this->extra_neighbor_data[graph_type] + this->extra_neighbor_offset[graph_type];

                // Copy existing neighbor data to new location with expanded layout
                for (int i = 0, j = 0; i < degree_a - 1; i++, j++) {
                    if (j % degree_a == degree_a - 1) { j++; }
                    ptr_new[j] = ptr_old[i];
                }
                ptr_new[degree_a - 1] = static_cast<weight_t>(b);
                ptr_new[2 * degree_a - 1] = weight;

                this->beg_ptr[graph_type][a] = ptr_new;

                weight_t *ptr_new_device = reinterpret_cast<weight_t *>(extra_neighbor_data_device[graph_type]) + extra_neighbor_offset[graph_type];
                this->beg_ptr_device_content[graph_type][a] = ptr_new_device;

                this->extra_neighbor_offset[graph_type] += degree_a * 2;
            }

        }




        void edgeDeletion(vertex_t a, vertex_t b, int graph_type){

            if (a > b) swap(a, b);
            long key = a * this->multiplier + b;

            if (this->graph_map[graph_type]->count(key) > 0) {

                index_t b_index_in_a = this->graph_map[graph_type]->at(key).first;
                index_t a_index_in_b = this->graph_map[graph_type]->at(key).second;


                this->graph_map[graph_type]->erase(key);

                int degree_a = --this->degree_list[graph_type][a];
                int degree_b = --this->degree_list[graph_type][b];

                assert(degree_a != 0);
                assert(degree_b != 0);
                this->edgeDeletionForNeighborData(a, b, degree_a, b_index_in_a, graph_type);
                this->edgeDeletionForNeighborData(b, a, degree_b, a_index_in_b, graph_type);

            } else {
                cerr << "Error: Edge does not exist in graph map: " << a << " " << b << endl;
                exit(EXIT_FAILURE);
            }

        }

        void edgeDeletionForNeighborData(vertex_t a, vertex_t b, int degree_a, index_t b_index_in_a, int graph_type){

            int bin_a = this->bin_size[graph_type][a]; // no need change bin size

            // if it's last edge in the vertex, don't need to update the neighbors_data
            if(degree_a != b_index_in_a){
                // else put the deleted edge to the last position, ignore it by --degree
                vertex_t a_last = this->beg_ptr[graph_type][a][degree_a];
                this->beg_ptr[graph_type][a][b_index_in_a] = a_last; // col
                this->beg_ptr[graph_type][a][b_index_in_a + bin_a] = this->beg_ptr[graph_type][a][degree_a + bin_a]; // weight

                if (a < a_last){
                    long key = a * this->multiplier + a_last;
                    this->graph_map[graph_type]->at(key).first = b_index_in_a;
                }
                else{
                    long key = a_last * this->multiplier + a;
                    this->graph_map[graph_type]->at(key).second = b_index_in_a;
                }

            }
        }


        void updateSparsiferFromResult(GPU_Stream_Edges & g_stream_edges){

            for (){
                
            }
        }

 
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
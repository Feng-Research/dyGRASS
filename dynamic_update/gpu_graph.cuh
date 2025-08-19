
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

struct ValueIndex {
      float value;
      int index;
};


class GPU_Stream_Edges{

    public:

        size_t max_capacity;
        bool overflow_flag;
        size_t  next_run_index;

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

            this->heuristic_sample_nodes = new vertex_t[max_capacity * 2];
        

            HRR(cudaMalloc((void **)&edges_device, sizeof(vertex_t)*max_capacity * 2));
            HRR(cudaMalloc((void **)&weights_device, sizeof(weight_t)*max_capacity));

            HRR(cudaMalloc((void **)&path_selected_device, sizeof(int)*max_capacity * n_steps));
            HRR(cudaMalloc((void **)&path_selected_flag_device, sizeof(int)*max_capacity));

            HRR(cudaMalloc((void **)&heuristic_sample_nodes_device, sizeof(vertex_t)*max_capacity * 2));

        }

        void loadEdgeFromStream(const EdgeStream& edge_stream){

            this->batch_size = edge_stream.batch_size;
            assert(batch_size == edge_stream.batch_edges.size());
            this->current_op = edge_stream.current_op;
            this->next_run_index = 0;

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

        bool next_run_exist(){
            return this->next_run_index < this->batch_size;
        }
        
        void loadEdgesToDevice(){
            this->heuristic_sample_num = 0;

            if (this->edges == nullptr || this->weights == nullptr){
                cout << "Error: edges or weights not loaded." << endl;
                std::cerr << "Edges not loaded to GPU stream." << std::endl;
                exit(EXIT_FAILURE);
                return;
            }

            if (this->batch_size - this->next_run_index > this->max_capacity){

                this->overflow_flag = true;
                this->load_size = this->max_capacity;
                
                
            }
            else{

                this->overflow_flag = false;
                this->load_size = this->batch_size - this->next_run_index;  
            }

            this->next_run_index += this->load_size;

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

        void heuristicRecovery(GPU_Dual_Graph * ggraph){
            if (this->heuristic_sample_num == 0) {
                cout << "Wrong call to heuristicRecovery()" << endl;
                return;
            } else {
                HRR(cudaMemcpy(heuristic_sample_nodes_device, heuristic_sample_nodes, sizeof(vertex_t) * heuristic_sample_num * 2, cudaMemcpyHostToDevice));
                // Launch heuristic recovery kernel here
                //TODO: launch heuristicRecoveryKernel<<<grid, block>>>(...);
                int size =  heuristic_sample_num * 2 * 2 * 16;
                assert(size  < load_size * n_steps);
                HRR(cudaMemcpy(path_selected_device, path_selected, sizeof(vertex_t) * size , cudaMemcpyHostToDevice));
                NBRW_heuristic_decremental<<<heuristic_sample_num * 2, 16>>>(ggraph, this);
                HRR(cudaMemcpy(path_selected_device, path_selected, sizeof(vertex_t) * size, cudaMemcpyDeviceToHost));
            }
        }


        ~GPU_Stream_Edges(){
            delete[] edges;
            delete[] weights;
            delete[] path_selected;
            delete[] path_selected_flag;
            delete[] heuristic_sample_nodes;
            

            cudaFree(edges_device);
            cudaFree(weights_device);
            cudaFree(path_selected_device);
            cudaFree(path_selected_flag_device);
            cudaFree(heuristic_sample_nodes_device);
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
        unordered_map<long, EdgeInfo>  * graph_map[2]; // Edge mapping for O(1) operations
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
                    b_index_in_a = this->graph_map[graph_type]->at(key).index_a;
                    a_index_in_b = this->graph_map[graph_type]->at(key).index_b;

                    this->graph_map[graph_type]->at(key).weight += added_weight;
                    this->beg_ptr[graph_type][a][b_index_in_a] += added_weight;
                    this->beg_ptr[graph_type][b][a_index_in_b] += added_weight;
                }
            }
        }

        void edgeInsertion(vertex_t a, vertex_t b, weight_t weight, int graph_type){

            if (a > b) swap(a, b);
            long key = a * this->multiplier + b;
            index_t degree_a, degree_b;

            if (this->graph_map[graph_type]->count(key) == 0){

                degree_a = ++this->degree_list[graph_type][a];
                degree_b = ++this->degree_list[graph_type][b];


                this->graph_map[graph_type]->emplace(key, EdgeInfo{degree_a - 1, degree_b - 1, weight});

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
                //check if extra memory is exhausted
                if (ptr_new + degree_a * 2 > this->extra_neighbor_data[graph_type] + this->edge_num[graph_type]*2){
                    cerr << "Error: Extra memory exhausted" << endl;
                    exit(EXIT_FAILURE);
                }

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

        void edgeInsertionForHeuristic(vertex_t a, vertex_t b){

            if (a > b) swap(a, b);
            long key = a * this->multiplier + b;

            
            if (this->graph_map[DENSE]->count(key) == 0){
                cerr << "Error: Edge not exist in dense graph, cannot insert for heuristic: " << a << " " << b << endl;
            } else{
                weight_t weight = this->graph_map[DENSE]->at(key).weight;

                if (this->graph_map[SPARSE]->count(key) == 0){
                     edgeInsertion(a, b, weight, SPARSE);
                }
                    
            }


        }


        void edgeDeletion(vertex_t a, vertex_t b, int graph_type){

            if (a > b) swap(a, b);
            long key = a * this->multiplier + b;

            if (this->graph_map[graph_type]->count(key) > 0) {

                index_t b_index_in_a = this->graph_map[graph_type]->at(key).index_a;
                index_t a_index_in_b = this->graph_map[graph_type]->at(key).index_b;


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
                    this->graph_map[graph_type]->at(key).index_a = b_index_in_a;
                }
                else{
                    long key = a_last * this->multiplier + a;
                    this->graph_map[graph_type]->at(key).index_b = b_index_in_a;
                }

            }
        }

        
        void updateSparsiferFromResult(GPU_Stream_Edges & g_stream_edges){

            int count = 0;
            int incremental_no_found_count = 0;
            int decremental_no_found_count = 0;

            g_stream_edges.downloadResult();
            int batch_size = g_stream_edges.load_size;
            int start_index = g_stream_edges.next_run_index - batch_size;
            if (g_stream_edges.current_op == INCREMENTAL){
                // for incremental, just add all found edges to sparse graph
                for (int i = 0; i < batch_size; i++){
                    if (g_stream_edges.path_selected_flag[i] != -1){
                        vertex_t a = g_stream_edges.edges[start_index + i * 2];
                        vertex_t b = g_stream_edges.edges[start_index + i * 2 + 1];
                        weight_t w = g_stream_edges.weights[start_index + i];
                        this->edgeInsertion(a, b, w, SPARSE);
                        count++;
                    }else{
                        incremental_no_found_count++;
                    }
                }
            } else if (g_stream_edges.current_op == DECREMENTAL){
                // for decremental, check the path_selected_flag, if 1, add the path to sparse graph
                for (int i = 0; i < batch_size; i++){
                    
                    vertex_t a = g_stream_edges.edges[start_index + i * 2];
                    vertex_t b = g_stream_edges.edges[start_index + i * 2 + 1];
                    int path_length = g_stream_edges.path_selected_flag[i];

                    if (path_length != 0){
                        
                        int path_start = i * g_stream_edges.n_steps;
                        vertex_t from = a;

                        for (int j = 0; j < path_length; j++){

                            vertex_t b = g_stream_edges.path_selected[path_start + j];
                            if ( a > b) swap(a, b);
                            long key = a * this->multiplier + b;

                            if (this->graph_map[SPARSE]->count(key) == 0){
                                count++;
                                weight_t weight = this->graph_map[DENSE]->at(key).weight;
                                this->edgeInsertion(a, b, weight, SPARSE);
                            }
                            
                            a = b;

                        }
                        
                    }else{

                        // int current_index = 
                        g_stream_edges.heuristic_sample_nodes[decremental_no_found_count * 2] = a;
                        g_stream_edges.heuristic_sample_nodes[decremental_no_found_count * 2 + 1] = b;

                        decremental_no_found_count++;

                    }
                }

            }

            g_stream_edges.heuristic_sample_num = decremental_no_found_count;
            cout << "Sparsifier updated with " << count << " edges from stream." <<  endl;
            if (count > 0){
                updateDeviceDualGraph();
            }

            if(decremental_no_found_count > 0){
                cout << "Not implemented yet" << endl;
                cout << "Heuristic recovery needed for " << decremental_no_found_count << " edges." <<  endl;
                g_stream_edges.heuristicRecovery(this);
                
                for (int i = 0; i < decremental_no_found_count * 2; i++){
                    
                    vertex_t a = g_stream_edges.heuristic_sample_nodes[i * 2];
                    vertex_t b = g_stream_edges.path_selected[i * 2];
                    vertex_t c = g_stream_edges.path_selected[i * 2 + 1];

                    this->edgeInsertionForHeuristic(a, b);
                    this->edgeInsertionForHeuristic(b, c);
                }
                    
            }

        }
        

 
       
        void sparse_map_to_1_based_mtx(string file_name){
            // std::ofstream txt_file(file_name, std::ios::out);
            // if(txt_file.is_open()){
            //     for(auto it = sparse_map->begin(); it != sparse_map->end(); ++it){

            //         auto key = it->first;
            //         long a = key / this->multiplier;
            //         long b = key % this->multiplier;
            //         txt_file << a + 1 << " " << b + 1 << " " << 1 << std::endl;
            //     }
            // }
            // cout<< "Updated sparse mtx file saved to: " << file_name << endl; 
                
        }


        ~GPU_Dual_Graph(){ 
            
        }
        
};


__global__ void
NBRW_decremental(
    GPU_Dual_Graph * G,
    GPU_Stream_Edges * stream_edges,
    int batch_size,
    int n_steps
){

    __shared__ ValueIndex sharedData[32];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState local_state;
    curand_init(tid, 10, 7, &local_state);

    unsigned int sourceIndex = stream_edges->edges_device[blockIdx.x * 2];
    unsigned int targetIndex = stream_edges->edges_device[blockIdx.x * 2 + 1];

    unsigned int currentVertex = sourceIndex;
    unsigned int previousVertex = 0;
    int targetFoundAt = -1; // TODO: make sure use same count as CPU side
    int step_count = 0;

    unsigned int path [max_steps];
    float total_R = 0;

    // firtst step don't need consider previous vertex
    unsigned int degree = G->degree_list_device[DENSE][currentVertex];
    unsigned int bin_size = G->bin_size_device[DENSE][currentVertex];
    weight_t * neighbors = G->beg_ptr_device[DENSE][currentVertex];
    unsigned int next = curand(&local_state) % degree;
    unsigned int nextVertex = static_cast<unsigned int>(neighbors[next]);
    float resistance = 1 / neighbors[bin_size + next];
    total_R += resistance;
    path[step_count] = nextVertex;
    

    if (nextVertex != targetIndex){

        previousVertex = currentVertex;
        currentVertex = nextVertex;
        step_count ++;

        while (step_count < n_steps){

            degree = G->degree_list_device[DENSE][currentVertex];
            if (degree == 1){
                break;
            }
            if (degree == 0){
                printf("Error: degree zero node encountered in NBRW_decremental\n");
                break;
            }

            bin_size = G->bin_size_device[DENSE][currentVertex];
            neighbors = G->beg_ptr_device[DENSE][currentVertex];
            next = curand(&local_state) % (degree - 1);
            nextVertex = static_cast<unsigned int>(neighbors[next]);

            if (nextVertex == previousVertex){
                nextVertex = static_cast<unsigned int>(neighbors[degree - 1]);
                resistance  = 1 / neighbors[bin_size + degree - 1];
            } else {
                resistance  = 1 / neighbors[bin_size + next];
            }

            path[step_count] = nextVertex;
            total_R += resistance;

            if (nextVertex == targetIndex){
                targetFoundAt = step_count;
                break;
            }

            previousVertex = currentVertex;
            currentVertex = nextVertex;
            step_count ++;
        }

    } else{

        targetFoundAt = step_count;
    }

    ValueIndex result;
    if (targetFoundAt != -1) {
        result = {total_R, (int)threadIdx.x};
    }else{
        result = {__FLT_MAX__, -1};
    }
    
    __syncthreads();

    ValueIndex minResult = blockReduceMin(result, sharedData);

    if (threadIdx.x == 0) {
       sharedData[0] = minResult;
    }

    __syncthreads();

    if (threadIdx.x == sharedData[0].index) {
        if (targetFoundAt != -1) {
            stream_edges->path_selected_flag_device[blockIdx.x] = targetFoundAt;

            for (int i = 0; i <= targetFoundAt; i++) {
                stream_edges->path_selected_device[blockIdx.x * n_steps + i] = path[i];
            }
        }else{
            stream_edges->path_selected_flag_device[blockIdx.x] = -1;
        }

    }

}

__global__ void
NBRW_incremental(
    GPU_Dual_Graph * G,
    GPU_Stream_Edges * stream_edges,
    int batch_size,
    float distortion
){  
    __shared__ ValueIndex sharedData[32];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState local_state;
    curand_init(tid, 10, 7, &local_state);

    unsigned int sourceIndex = stream_edges->edges_device[blockIdx.x * 2];
    unsigned int targetIndex = stream_edges->edges_device[blockIdx.x * 2 + 1];
    float edge_weight = stream_edges->weights_device[blockIdx.x];

    unsigned int currentVertex = sourceIndex;
    unsigned int previousVertex = 0;
    int targetFoundAt = -1;
    int step_count = 0;

    float total_R = 0;

    // first neighbor
    unsigned int degree = G->degree_list_device[SPARSE][currentVertex];
    unsigned int bin_size = G->bin_size_device[SPARSE][currentVertex];
    weight_t * neighbors = G->beg_ptr_device[SPARSE][currentVertex];
    unsigned int next = curand(&local_state) % degree;
    unsigned int nextVertex = static_cast<unsigned int>(neighbors[next]);
    float resistance = 1 / neighbors[bin_size + next];
    total_R += resistance;

    if (nextVertex != targetIndex){

        previousVertex = currentVertex;
        currentVertex = nextVertex;
        step_count ++;

        while (step_count < max_steps){

            degree = G->degree_list_device[SPARSE][currentVertex];
            if (degree == 1){
                break;
            }
            if (degree == 0){
                printf("Error: degree zero node encountered in NBRW_incremental\n");
                break;
            }

            bin_size = G->bin_size_device[SPARSE][currentVertex];
            neighbors = G->beg_ptr_device[SPARSE][currentVertex];
            next = curand(&local_state) % (degree - 1);
            nextVertex = static_cast<unsigned int>(neighbors[next]);

            if (nextVertex == previousVertex){
                nextVertex = static_cast<unsigned int>(neighbors[degree - 1]);
                resistance  = 1 / neighbors[bin_size + degree - 1];
            } else {
                resistance  = 1 / neighbors[bin_size + next];
            }

            total_R += resistance;

            //Termination conditions
            if (total_R * edge_weight >= distortion) {
                break;
            }

            if (nextVertex == targetIndex){
                targetFoundAt = step_count;
                break;
            }

            previousVertex = currentVertex;
            currentVertex = nextVertex;
            step_count ++;
        }

    }else{
        targetFoundAt = step_count;
    }

    ValueIndex result;
    if (targetFoundAt != -1) {
        result = {total_R, (int)threadIdx.x};
    }else{
        result = {__FLT_MAX__, -1};
    }

    __syncthreads();

    ValueIndex minResult = blockReduceMin(result, sharedData);

    if (threadIdx.x == 0) {
       sharedData[0] = minResult;
    }

    __syncthreads();

    if (threadIdx.x == sharedData[0].index) {
        if (targetFoundAt != -1) {
            stream_edges->path_selected_flag_device[blockIdx.x] = targetFoundAt;

        }else{
            stream_edges->path_selected_flag_device[blockIdx.x] = -1;
        }

    }

}


__global__ void
NBRW_heuristic_decremental(
    GPU_Dual_Graph * G,
    GPU_Stream_Edges * stream_edges
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState local_state;
    curand_init(tid, 10, 7, &local_state);

    unsigned int sourceIndex = stream_edges->heuristic_sample_nodes_device[tid * 2];
    unsigned int currentVertex = sourceIndex;
    unsigned int previousVertex = 0;
    unsigned int path[2];

    unsigned int degree = G->degree_list_device[DENSE][currentVertex];
    weight_t * neighbors = G->beg_ptr_device[DENSE][currentVertex];
    unsigned int next  = curand(&local_state) % degree;
    unsigned int nextVertex = static_cast<unsigned int>(neighbors[next]);
    path[0] = nextVertex;

    previousVertex = currentVertex;
    currentVertex = nextVertex;

    degree = G->degree_list_device[DENSE][currentVertex];
    next = curand(&local_state) % (degree -1);
    nextVertex = static_cast<unsigned int>(neighbors[next]);
    
    if (nextVertex == previousVertex) {
        nextVertex = static_cast<unsigned int>(neighbors[degree - 1]);
    }
    path[2] = nextVertex;

    for (int i = 0; i < 2; i++){
        stream_edges->path_selected_device[tid * 2 + i] = path[i];
    }

}


__device__ ValueIndex reduceMin(ValueIndex a, ValueIndex b) {
    return (a.value < b.value) ? a : b;  // Find minimum value
}

__device__ ValueIndex warpReduceMin(ValueIndex val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        ValueIndex other;
        other.value = __shfl_down_sync(0xffffffff, val.value, offset);
        other.index = __shfl_down_sync(0xffffffff, val.index, offset);
        val = reduceMin(val, other);  // Changed to reduceMin
    }
    return val;
}

__device__ ValueIndex blockReduceMin(ValueIndex val, ValueIndex* sharedData) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    // Step 1: Warp-level reduction
    val = warpReduceMin(val);  // Changed to warpReduceMin

    // Step 2: Store warp results?
    if (lane == 0) {
        sharedData[warp] = val;
    }
    __syncthreads();

    // Step 3: Final reduction among warps
    if (warp == 0) {  // Only first warp
          if (threadIdx.x < blockDim.x / 32) {
              val = sharedData[threadIdx.x];  // Load valid data
          } else {
              val = {__FLT_MAX__, -1};        // Safe initialization
          }
          val = warpReduceMin(val);  // Now safe - all threads have valid data
      }

    return val; // Thread 0 has minimum value with corresponding index and step
}

#endif
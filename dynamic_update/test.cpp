/**
 * Test program for auto_detect_graph_info function
 * 
 * This program creates various MTX file formats and tests the
 * auto-detection capabilities of the graph info function.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "functions.h"

using namespace std;
namespace fs = std::filesystem;

// Helper function to create test MTX files
void create_test_file(const string& filename, const string& content) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot create test file " << filename << endl;
        return;
    }
    file << content;
    file.close();
    cout << "Created test file: " << filename << endl;
}

// Helper function to print GraphInfo results
void print_graph_info(const string& filename, const GraphInfo& info) {
    cout << "\n=== Results for " << filename << " ===" << endl;
    cout << "Skip lines: " << info.skip_lines << endl;
    cout << "Has MatrixMarket header: " << (info.has_matrixmarket_header ? "Yes" : "No") << endl;
    cout << "Has dimensions line: " << (info.has_dimensions_line ? "Yes" : "No") << endl;
    cout << "Is weighted: " << (info.is_weighted ? "Yes" : "No") << endl;
    cout << "Is Laplacian: " << (info.is_laplacian ? "Yes" : "No") << endl;
    cout << "Is triangular: " << (info.is_triangle ? "Yes" : "No") << endl;
    cout << "Index base: " << info.base << "-based" << endl;
    cout << "Vertex range: [" << info.v_min << ", " << info.v_max << "]" << endl;
    cout << "===========================================" << endl;
}

// Test cases
void run_tests() {
    // Create test directory
    string test_dir = "test_mtx_files";
    if (!fs::exists(test_dir)) {
        fs::create_directory(test_dir);
    }
    
    cout << "Creating test MTX files..." << endl;
    
    // Test 1: Standard MTX with header and dimensions (1-based, weighted, adjacency)
    create_test_file(test_dir + "/test1_standard.mtx", 
        "%%MatrixMarket matrix coordinate real general\n"
        "% This is a test adjacency matrix\n"
        "% 4 vertices, 6 edges\n"
        "4 4 6\n"
        "1 2 0.5\n"
        "1 3 1.0\n"
        "2 3 0.8\n"
        "2 4 1.2\n"
        "3 4 0.9\n"
        "4 1 0.7\n"
    );
    
    // Test 2: MTX with header and dimensions (1-based, weighted, symmetric/triangular)
    create_test_file(test_dir + "/test2_triangular.mtx",
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "% Symmetric matrix - only upper triangular stored\n"
        "4 4 4\n"
        "1 2 0.5\n"
        "1 3 1.0\n"
        "2 3 0.8\n"
        "3 4 0.9\n"
    );
    
    // Test 3: MTX with header but no dimensions (1-based, weighted)
    create_test_file(test_dir + "/test3_no_dimensions.mtx",
        "%%MatrixMarket matrix coordinate real general\n"
        "% No dimensions line\n"
        "1 2 0.5\n"
        "2 3 1.0\n"
        "3 4 0.8\n"
        "4 1 1.2\n"
    );
    
    // Test 4: No header, with dimensions (1-based, weighted)
    create_test_file(test_dir + "/test4_dimensions_only.mtx",
        "4 4 4\n"
        "1 2 0.5\n"
        "2 3 1.0\n"
        "3 4 0.8\n"
        "4 1 1.2\n"
    );
    
    // Test 5: No header, no dimensions (1-based, weighted)
    create_test_file(test_dir + "/test5_raw_data.mtx",
        "1 2 0.5\n"
        "2 3 1.0\n"
        "3 4 0.8\n"
        "4 1 1.2\n"
    );
    
    // Test 6: 0-based indexing (no header, weighted)
    create_test_file(test_dir + "/test6_zero_based.mtx",
        "0 1 0.5\n"
        "1 2 1.0\n"
        "2 3 0.8\n"
        "3 0 1.2\n"
    );
    
    // Test 7: Unweighted adjacency matrix (1-based)
    create_test_file(test_dir + "/test7_unweighted.mtx",
        "%%MatrixMarket matrix coordinate pattern general\n"
        "4 4 4\n"
        "1 2\n"
        "2 3\n"
        "3 4\n"
        "4 1\n"
    );
    
    // Test 8: Laplacian matrix (negative weights)
    create_test_file(test_dir + "/test8_laplacian.mtx",
        "%%MatrixMarket matrix coordinate real general\n"
        "4 4 10\n"
        "1 1 2.0\n"
        "1 2 -0.5\n"
        "1 4 -1.5\n"
        "2 2 1.3\n"
        "2 1 -0.5\n"
        "2 3 -0.8\n"
        "3 3 1.7\n"
        "3 2 -0.8\n"
        "3 4 -0.9\n"
        "4 4 3.4\n"
    );
    
    // Test 9: Edge case - single edge
    create_test_file(test_dir + "/test9_single_edge.mtx",
        "2 2 1\n"
        "1 2 1.0\n"
    );
    
    // Test 10: Large vertex IDs (test dimension detection)
    create_test_file(test_dir + "/test10_large_ids.mtx",
        "1000 1000 3\n"
        "1 500 0.5\n"
        "500 1000 1.0\n"
        "1000 1 0.8\n"
    );
    
    // Test 11: Full adjacency matrix (complete graph, non-triangular)
    create_test_file(test_dir + "/test11_full_adjacency.mtx",
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 6\n"
        "1 2 1.0\n"
        "1 3 1.0\n"
        "2 1 1.0\n"
        "2 3 1.0\n"
        "3 1 1.0\n"
        "3 2 1.0\n"
    );
    
    // Test 12: Full Laplacian matrix (with diagonal elements)
    create_test_file(test_dir + "/test12_full_laplacian.mtx",
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 3\n"
        "1 1 2.0\n"
        "1 2 -1.0\n"
        "1 3 -1.0\n"
        "2 1 -1.0\n"
        "2 2 2.0\n"
        "2 3 -1.0\n"
        "3 1 -1.0\n"
        "3 2 -1.0\n"
        "3 3 2.0\n"
    );
    
    cout << "\nRunning tests..." << endl;
    
    // Test all files
    vector<string> test_files = {
        "test1_standard.mtx",
        "test2_triangular.mtx", 
        "test3_no_dimensions.mtx",
        "test4_dimensions_only.mtx",
        "test5_raw_data.mtx",
        "test6_zero_based.mtx",
        "test7_unweighted.mtx",
        "test8_laplacian.mtx",
        "test9_single_edge.mtx",
        "test10_large_ids.mtx",
        "test11_full_adjacency.mtx",
        "test12_full_laplacian.mtx"
    };
    
    for (const string& filename : test_files) {
        string full_path = test_dir + "/" + filename;
        try {
            GraphInfo info = auto_detect_graph_info(full_path.c_str());
            print_graph_info(filename, info);
        } catch (const exception& e) {
            cout << "Error testing " << filename << ": " << e.what() << endl;
        }
    }
}

// Interactive test mode
void interactive_test() {
    cout << "\n=== Interactive Test Mode ===" << endl;
    cout << "Enter MTX filename to test (or 'quit' to exit): ";
    
    string filename;
    while (cin >> filename && filename != "quit") {
        if (!fs::exists(filename)) {
            cout << "File '" << filename << "' does not exist." << endl;
        } else {
            try {
                GraphInfo info = auto_detect_graph_info(filename.c_str());
                print_graph_info(filename, info);
            } catch (const exception& e) {
                cout << "Error: " << e.what() << endl;
            }
        }
        cout << "\nEnter next filename (or 'quit' to exit): ";
    }
}

int main(int argc, char* argv[]) {
    cout << "dyGRASS Graph Info Auto-Detection Test Program" << endl;
    cout << "===============================================" << endl;
    
    if (argc > 1) {
        // Test specific file provided as argument
        string filename = argv[1];
        cout << "Testing file: " << filename << endl;
        
        if (!fs::exists(filename)) {
            cout << "Error: File '" << filename << "' does not exist." << endl;
            return 1;
        }
        
        try {
            GraphInfo info = auto_detect_graph_info(filename.c_str());
            print_graph_info(filename, info);
        } catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
            return 1;
        }
    } else {
        // Run automated tests
        char choice;
        cout << "Choose test mode:" << endl;
        cout << "[A] Automated tests (create and test various MTX formats)" << endl;
        cout << "[I] Interactive mode (test your own files)" << endl;
        cout << "[B] Both" << endl;
        cout << "Enter choice (A/I/B): ";
        cin >> choice;
        
        choice = tolower(choice);
        
        if (choice == 'a' || choice == 'b') {
            run_tests();
        }
        
        if (choice == 'i' || choice == 'b') {
            interactive_test();
        }
    }
    
    cout << "\nTest completed!" << endl;
    
    // Clean up test files
    cout << "Clean up test files? (y/n): ";
    char cleanup;
    cin >> cleanup;
    if (tolower(cleanup) == 'y') {
        try {
            fs::remove_all("test_mtx_files");
            cout << "Test files cleaned up." << endl;
        } catch (const exception& e) {
            cout << "Error cleaning up: " << e.what() << endl;
        }
    }
    
    return 0;
}
using Laplacians
using SparseArrays
using LinearAlgebra
using Arpack
using JLD2
using IterativeSolvers
using Statistics


struct CholPrecond
    F
end


function CliqueW(ar, W)

    mx = mx_func(ar)

    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    vv = zeros(Float32, 0)
    for i =1:length(ar)
        nd = sort(ar[i])

        append!(rr, nd[1])
        append!(cc, nd[2])
        append!(vv, W[i])

    end

    #vv = ones(Int, length(rr))

    mat1 = sparse(rr,cc,vv, mx, mx)

    return mat2 = mat1 + sparse(mat1')

end

function mx_func(ar)

    mx2 = Int(0)
    aa = Int(0)

    for i =1:length(ar)

    	mx2 = max(aa, maximum(ar[i]))
    	aa = mx2

    end
    return mx2

end

function create_laplacian(ar, W; offset = true)
    A = CliqueW(ar, W)
    L = lap(A)
    if offset
        for ii = 1:size(A, 1)
            L[ii, ii] = L[ii, ii] + 1e-6
        end
    end
    
    return L
end

function create_laplacian(A; offset = true)
    L = lap(A)
    if offset
        for ii = 1:size(A, 1)
            L[ii, ii] = L[ii, ii] + 1e-6
        end
    end
    
    return L
end

function readMtx(Inp; base = 0, type = "adj", weighted = true, sort = true)
    # the output should be 1-based as Julia is 1-based
    # type can be adj or lap
    println("Reading ", Inp, " with base ", base, " and type ", type, " and weighted ", weighted)
    io = open(Inp, "r")
    
   
    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    w = zeros(Float64, 0)
    edges = []
    
    if base == 0
        base = 1
    elseif base == 1
        base = 0
    else
        error("Base should be 0 or 1")
        
    end

    while !eof(io)
        ln = readline(io)
        sp = split(ln)

        r = parse(Int, sp[1]) + base
        c = parse(Int, sp[2]) + base
        if r == c
            if type == "adj"
                error("Self loop find in adjacency matrix")
            elseif type == "lap"
                continue
            end
        end
        if r > c
            r, c = c, r
        end
        push!(rr, r)
        push!(cc, c)

        if weighted
            v = parse(Float64, sp[3])
            v = type == "lap" ? -v : v
            append!(w, v)
        else
            # it must be a adjacency matrix without weights
            append!(w, 1.0)  # default weight is 1.0 for unweighted graphs
        end

    end

    if sort
        A = sparse(rr, cc, w)
        fdnz = findnz(A)

        rr = fdnz[1]
        cc = fdnz[2]
        w = fdnz[3]

        for i = 1:length(rr)
            push!(edges, [rr[i], cc[i]])
        end
    else
        for i = eachindex(rr)
            push!(edges, [rr[i], cc[i]])
        end
    end



    return edges, w

end


function run_cmd(graph_name, distortion = 100, walker = 512)
    cmd = `./incremental_update`
    arg1 = graph_name
    arg2 = distortion
    arg3 = walker # walker

    cmd = `$cmd $arg1 $arg2 $arg3`
    # run(cmd)
    output = read(cmd, String)
    return output
end

function check_density(ar, V_size)
    return length(ar) / V_size
end

function V_size(ar)
    mx = mx_func(ar)
    return mx
end

function checkBase(file_path)
    """
    Determines if a graph file uses 0-based or 1-based vertex indexing by examining all vertices.
    
    Args:
        file_path: Path to the graph file (MTX format or edge list)
    
    Returns:
        0 if 0-based indexing (minimum vertex index is 0)
        1 if 1-based indexing (minimum vertex index is 1)
        nothing if file doesn't exist or is empty
    """
    if !isfile(file_path)
        println("File $file_path does not exist.")
        return nothing
    end

    min_vertex = typemax(Int)  # Start with maximum possible integer
    line_count = 0
    
    try
        open(file_path, "r") do f
            for line in eachline(f)
                # Skip comment lines (starting with %)
                if startswith(strip(line), "%") || isempty(strip(line))
                    continue
                end
                
                line_count += 1
                parts = split(strip(line))
                
                # Handle both 2-column (unweighted) and 3-column (weighted) formats
                if length(parts) >= 2
                    try
                        # Parse first two columns as vertex indices
                        v1 = parse(Int, parts[1])
                        v2 = parse(Int, parts[2])
                        
                        # Update minimum vertex index
                        min_vertex = min(min_vertex, v1, v2)
                        
                        # Early termination: if we find 0, we know it's 0-based
                        if min_vertex == 0
                            return 0
                        end
                    catch ArgumentError
                        # Skip lines that don't contain valid integers
                        continue
                    end
                end
            end
        end
        
        # If no valid lines were processed
        if line_count == 0 || min_vertex == typemax(Int)
            println("Warning: No valid edge data found in file $file_path")
            return nothing
        end
        
        # Return based on minimum vertex found
        if min_vertex == 0
            return 0  # 0-based indexing
        elseif min_vertex == 1
            return 1  # 1-based indexing
        else
            println("Warning: Minimum vertex index is $min_vertex, expected 0 or 1")
            return min_vertex >= 1 ? 1 : 0
        end
        
    catch e
        println("Error reading file $file_path: $e")
        return nothing
    end
end

function checkMtxProp(file_path)
    if !isfile(file_path)
        println("File $file_path does not exist.")
        return false
    end

    line_data = open(file_path, "r") do f
        split(readline(f))
    end

    if length(line_data) == 2
        return ("adj", false)
    end

    if length(line_data) == 3
        if line_data[1] == line_data[2]
            println("Warning: Treat as lap, but maybe is a self-loop in adjacency matrix.")
            return ("lap", true)  # treat as adjacency if self-loop found
        else
            w = parse(Float64, line_data[3])
            if w < 0
                return ("lap", true)  # treat as lap if negative weights found
            else
                return ("adj", true)  # treat as adjacency if no self-loop and no negative weights
            end
        end
    end

end

# incremental
function incremental_test(graph_name; 
    initial = false, 
    exp = false, 
    distortion = 100
    )
    CND_list = []
    density_list = []
    output_list = []
    
    if !isdir("../dataset/" * graph_name)
        # println("Note: dataset folder should be in the current directory.")
        error("Dataset folder for $graph_name does not exist.")
    end

    original_sparse = "../dataset/" * graph_name * "/adj_sparse.mtx"
    original_edges, original_weights = readMtx(original_sparse, base = 0, type = "adj", weighted = true)

    
    original_dense = "../dataset/" * graph_name * "/dense.mtx"
    g_type, is_weighted = checkMtxProp(original_dense)
    dense_edges, dense_weights = readMtx(original_dense, base = 1, type = g_type, weighted = is_weighted)

    
    added_edges = "../dataset/" * graph_name * "/ext.mtx"
    added_edges, added_weights = readMtx(added_edges, base=1, type="adj", weighted=true)

    number_of_vertices = V_size(original_edges)
    println("Number of vertices = ", number_of_vertices)
    density = check_density(original_edges, number_of_vertices)
    println("Original sparsifier density = ", density)
    L_sparse_original = create_laplacian(original_edges, original_weights)
    L_dense_original = create_laplacian(dense_edges, dense_weights)

    if initial # original CND and density
        CND = eigs(L_dense_original, L_sparse_original)
        println("Original sparsifier CND = ", maximum(real(CND[1])))

        append!(CND_list, maximum(real(CND[1])))
        append!(density_list, density)
        append!(output_list, "Original CND")
    end
    

    

    append!(dense_edges, added_edges)
    append!(dense_weights, added_weights)

    append!(original_edges, added_edges)
    append!(original_weights, added_weights)

    L_dense_updated = create_laplacian(dense_edges, dense_weights)
    L_sparse_fully_updated = create_laplacian(original_edges, original_weights)

    if initial # sparsifier without update
        CND = eigs(L_dense_updated, L_sparse_original)
        println("No updated sparsifier CND = ", maximum(real(CND[1])))
        append!(CND_list, maximum(real(CND[1])))
        append!(density_list, density)
        append!(output_list, "No updated sparsifier")
    end

    density = check_density(original_edges, number_of_vertices)
    println("Fully updated sparsifier density = ", density)

    if initial #sparsifier fully updated
        CND = eigs(L_dense_updated, L_sparse_fully_updated)
        println("Fully updated sparsifier CND = ", maximum(real(CND[1])))
        append!(CND_list, maximum(real(CND[1])))
        append!(density_list, density)
        append!(output_list, "Fully updated sparsifier")
    end
    #################################################################################
      
    if exp

        println("distortion shreshold: ", distortion)
        output = run_cmd(graph_name, distortion, 512)
        # println(output)
        updated_sparse = "../dataset/" * graph_name * "/updated_adj_sparse.mtx"
        updated_edges, updated_weights = readMtx(updated_sparse, base=1, type="adj", weighted=true)
        L_sparse_updated = create_laplacian(updated_edges, updated_weights)
        CND = eigs(L_dense_updated, L_sparse_updated)
        append!(CND_list, maximum(real(CND[1])))
        append!(density_list, check_density(updated_edges, number_of_vertices))
        push!(output_list, output)

    end

    return CND_list, density_list, output_list

end

function saveEXT(ar, W, path; base_zero =false)


    if ! base_zero
        open(path, "w") do file
            for (x, c) in zip(ar, W)
                a, b = x
                println(file, "$a $b $c")
            end
        end
    else
        println("Saving edges with 0-based indexing")
        open(path, "w") do file
            for (x, c) in zip(ar, W)
                a, b = x
                a -= 1
                b -= 1
                println(file, "$a $b $c")
            end
        end
    end


    println("File saved to ", path)
end

function saveDEL(ar, path)
    open(path, "w") do file
        for x in ar
            a, b = x
            println(file, "$a $b")
        end
    end
    println("File saved to ", path)
end

function prepareDecremental(graph_name)

    if !isdir("../dataset/" * graph_name)
        error("Dataset folder for $graph_name does not exist.")
    end


    original_dense = "../dataset/" * graph_name * "/dense.mtx"
    g_type, is_weighted = checkMtxProp(original_dense)
    dense_edges, dense_weights = readMtx(original_dense, base = 1, type = g_type, weighted = is_weighted)
    
    added_edges = "../dataset/" * graph_name * "/ext.mtx"
    added_edges, added_weights = readMtx(added_edges, base=1, type="adj", weighted=true)

    append!(dense_edges, added_edges)
    append!(dense_weights, added_weights)

    saveEXT(dense_edges, dense_weights, "../dataset/" * graph_name * "/updated_dense.mtx")

end

function makeSureAdjOneBase(folder, file_name, new_file_name)
    if !isdir(folder)
        error("Folder $folder does not exist.")
    end

    original_file = folder * "/" * file_name
    g_type, is_weighted = checkMtxProp(original_file)
    base = checkBase(original_file)
    edges, weights = readMtx(original_file, base = base, type = g_type, weighted = is_weighted)
    
    @assert minimum(Iterators.flatten(edges)) == 1
    saveEXT(edges, weights, folder * "/" * new_file_name; base_zero = false)
end

function read_del_edges(graph_name; percentage = 1.0)
    file_name = string("../del_edge/", string(percentage), "/",graph_name, "_del_edge_local.jld2")
    try
        @load file_name ext Nde
        return ext, Nde
    catch e
        println("Error: ", e)
    end

    
end

function run_decremental(graph_name)
    #TODO : implement percentage selections
    cmd = `./decremental_update`
    arg1 = graph_name


    cmd = `$cmd $arg1`
    # run(cmd)
    output = read(cmd, String)
    return output
end

function decremental_test(graph_name; percentage= 1.0)
    # incremental added edges
    added_edges = "../dataset/" * graph_name * "/ext.mtx"
    added_edges, added_weights = readMtx(added_edges, base=1, type="adj", weighted=true)
    
    # decremental deledted edges, Nde is for delete convenience
    del, Nde = read_del_edges(graph_name)

    # original edense graph
    original_dense = "../dataset/" * graph_name * "/dense.mtx"
    g_type, is_weighted = checkMtxProp(original_dense)
    dense_edges, dense_weights = readMtx(original_dense, base = 1, type = g_type, weighted = is_weighted)
    append!(dense_edges, added_edges)
    append!(dense_weights, added_weights)
    saveEXT(dense_edges, dense_weights, "../dataset/" * graph_name * "/updated_dense.mtx")

    dictG = Dict(key => dense_weights[i] for (i, key) in enumerate(dense_edges))

    # original_dense = "../dataset/" * graph_name * "/dense.mtx"
    # g_type, is_weighted = checkMtxProp(original_dense)
    # ### sort = false !!! to maintain the idnex
    # dense_edges, dense_weights = readMtx(original_dense, base = 1, type = g_type, weighted = is_weighted, sort=false)

    # create incremental, decremental updated dense graph
    for i in eachindex(del)
        del_edge = del[i]
        if haskey(dictG, del_edge)
            delete!(dictG, del_edge)
        else
            println("Edge ", del_edge, " not exists in dense graph, edge index: ", i)   
        end
    end

    dense_edges = collect(keys(dictG))
    dense_weights = collect(values(dictG))

    L_dense = create_laplacian(dense_edges, dense_weights)
    dictG = Dict(key => dense_weights[i] for (i, key) in enumerate(dense_edges))

    #### make sure same del edges
    saveDEL(del,"../dataset/" * graph_name * "/del.mtx")

    output = run_decremental(graph_name)
    # println(output)

    # decremental newly introduced edges
    de_added_edge = "../dataset/" * graph_name * "/added_edges.mtx"
    de_added_edge, _ = readMtx(de_added_edge, base=1, type="adj", weighted=false)

    # incremental updated sparsifier
    updated_sparse = "../dataset/" * graph_name * "/updated_adj_sparse.mtx"
    updated_edges, updated_weights = readMtx(updated_sparse, base=1, type="adj", weighted=true)
    dictSP = Dict(key => updated_weights[i] for (i, key) in enumerate(updated_edges))
    # delete decremntal edges

    for i = eachindex(del)
        if haskey(dictSP, del[i])
            delete!(dictSP, del[i])
        else
            println("Edge not exists in SP: ", del[i], "index: ", i)   
        end
    end

    updated_edges = collect(keys(dictSP))
    updated_weights = collect(values(dictSP))
    # append new edges
    for i = eachindex(de_added_edge)
        if haskey(dictG, de_added_edge[i])
            push!(updated_edges, de_added_edge[i])
            append!(updated_weights, dictG[de_added_edge[i]])
        else
            println("Edge not exists in G: ", de_added_edge[i], "index: ", i)   
        end
    end

    L_sparse = create_laplacian(updated_edges, updated_weights)
    CND = eigs(L_dense, L_sparse)
    println("CND: ", maximum(real(CND[1])))

    density = check_density(updated_edges, size(L_sparse, 1))
    println("Sparsifier density: ", density)
    return maximum(real(CND[1])), density, output
end


#TODO: find a better way do this, define it  in here may cause a issue for other package
function LinearAlgebra.ldiv!(y::AbstractVector, P::CholPrecond, x::AbstractVector)
    copyto!(y, P.F \ x)
    return y
end

function pcg_test(A, B)
    # A is dense matriix, B is conditioner matrix
    n = size(A, 1)
    b = randn(n)
    b .-= mean(b);  # to avoid the null space (all-ones vector)

    x = randn(n)

    # Flexible struct to hold any factor type
    

    

    Pl = CholPrecond(cholesky(B))


    # Run CG
    x, history = cg!(x, A, b; Pl=Pl, reltol=1e-8, log=true)
    println("Final residual estimate: ", history[:resnorm][end])



    r = b - A * x # should be close to zero
    resnorm = norm(r)           # residual norm (2-norm)
    relres = norm(r) / norm(b)  # relative residual
    iter_count = length(history[:resnorm])
    println("Residual norm: ", resnorm)
    println("Relative residual: ", relres)
    println("Iteration count: ", iter_count)
    return  iter_count
end


function preProcessingForGRASS(graph_input, graph_output)

    g_type, is_weighted = checkMtxProp(graph_input)
    edges, weights = readMtx(graph_input, base = 1, type = g_type, weighted = is_weighted)
    L = create_laplacian(edges, weights; offset = false)
    matrix_size = size(L, 1)
    L_tril = tril(L)  # lower triangular part
    fdnz = findnz(L_tril)
    rr = fdnz[1]
    cc = fdnz[2]
    ww = fdnz[3]
    edge_num = length(rr)

    open(graph_output, "w") do file
        println(file, matrix_size, " ", matrix_size, " ", edge_num)
        for i in 1:edge_num
            println(file, rr[i], " ", cc[i], " ", ww[i])
        end
    end
    println("Preprocessing complete. Output saved to ", graph_output)

end

function postProcessGRASS(graph_path; fraction = false)

    io = open(graph_path, "r")
    readline(io)  # Skip the first line

    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    ww = zeros(Float64, 0)
    ff = zeros(Float64, 0)
    edges = []

    while  !eof(io)
        line = readline(io)
        sp = split(line)

        r = parse(Int, sp[1])
        c = parse(Int, sp[2])
        w = parse(Float64, sp[3])

        if fraction
            f = parse(Float64, sp[4])
        else
            f = 1.0
        end

        push!(rr, r)
        push!(cc, c)
        push!(ww, w)
        push!(ff, f)

    end

    L = sparse(rr, cc, ww./ff)
    L1 = tril(L,-1).*-1
    # A = L1 + sparse(L1')

    fdnz = findnz(L1)
    rr = fdnz[1]
    cc = fdnz[2]
    ww = fdnz[3]

    edges = Any[]
    for i in eachindex(rr)
        push!(edges, [rr[i], cc[i]])
    end

    return edges, ww

end

function run_GRASS(graph_name; CND = 100)
    # run it in random_walk_incremental_2
    dataset_folder = "../grass"
    entries = readdir(dataset_folder; join=true)
    subfolders = filter(isdir, entries)
    dataset_names = basename.(subfolders)

    
    if !(graph_name in dataset_names)
        graph_input = "../dataset/" * graph_name * "/dense.mtx"
        graph_output = dataset_folder * "/" * graph_name * "/L_tril.mtx"
        if !isdir(dataset_folder * "/" * graph_name)
            mkpath(dataset_folder * "/" * graph_name)
        end
        println("Dataset folder for $graph_name does not exist in $dataset_folder. Now Preprocessing from $graph_input to $graph_output")
        preProcessingForGRASS(graph_input, graph_output)
    end


    CND_folder = dataset_folder * "/" * graph_name * "/" * string(CND)
    if !isdir(CND_folder)
        mkpath(CND_folder)
        curernt_dir = pwd()
        cd(CND_folder)
        print(pwd())
        graph_file= "../L_tril.mtx"
        cmd = `grass-v1.0 -m $graph_file -c $CND`
        output = read(cmd, String)

        println("Output: ", output)
        sleep(1)
        Gmat_path = "Gmat.mtx"
        Pmat_path = "Pmat.mtx"
        edges, w = postProcessGRASS(Gmat_path; fraction = false)
        saveEXT(edges, w, "Gmat_adj_0.mtx"; base_zero = true)
        saveEXT(edges, w, "Gmat_adj.mtx"; base_zero = false)

        edges, w = postProcessGRASS(Pmat_path; fraction = true)
        saveEXT(edges, w, "Pmat_adj_0.mtx"; base_zero = true)
        saveEXT(edges, w, "Pmat_adj.mtx"; base_zero = false)

        pattern = r"Final eig_max:\s*[\d\.]+;\s*eig_min:\s*[\d\.]+\s*;\s*condNum:\s*([\d\.]+)"
        m = match(pattern, output)

        if m !== nothing
            cond_num = parse(Float64, m.captures[1])
            println("Extracted condNum: ", cond_num)
        else
            println("condNum not found")
        end

        open("final_CND.txt", "w") do file
            println(file, cond_num)
        end

        cd(curernt_dir)
    else
        println("CND folder for $graph_name with CND $CND already exists. No need to run GRASS again.")
    end

end


function iteration_exp(graph_name, CND_list)

    len = length(CND_list)

    added_edges = "../dataset/" * graph_name * "/ext.mtx"
    added_edges, added_weights = readMtx(added_edges, base=1, type="adj", weighted=true)

    iter_num = Any[]
    for i in 1:len
        run_GRASS(graph_name, CND = CND_list[i])
        # the file is ready for iteration test
        original_dense = "../grass/" * graph_name * "/" * string(CND_list[i]) * "/Gmat_adj.mtx"
        dense_edges, dense_weights = readMtx(original_dense, base = 1, type = "adj", weighted = true)
        original_sparse = "../grass/" * graph_name * "/" * string(CND_list[i]) * "/Pmat_adj.mtx"
        sparse_edges, sparse_weights = readMtx(original_sparse, base = 1, type = "adj", weighted = true)

        L_dense_original = create_laplacian(dense_edges, dense_weights)
        L_sparse_original = create_laplacian(sparse_edges, sparse_weights)

        iter_orginal = pcg_test(L_dense_original, L_sparse_original)

        dictG = Dict(key => dense_weights[i] for (i, key) in enumerate(dense_edges))

        filtered_added_edges = []
        filtered_added_weights = []
        for i in eachindex(added_edges)
            if ! haskey(dictG, added_edges[i])
                push!(filtered_added_edges, added_edges[i])
                push!(filtered_added_weights, added_weights[i])
            else
                println("Edge alreday exists in dense at first: ", added_edges[i], "index: ", i)   
            end
        end

        saveEXT(filtered_added_edges, filtered_added_weights, "../grass/" * graph_name * "/" * string(CND_list[i]) * "/ext.mtx")

        append!(dense_edges, filtered_added_edges)
        append!(dense_weights, filtered_added_weights)

        L_dense_updated = create_laplacian(dense_edges, dense_weights)

        iter_updated = pcg_test(L_dense_updated, L_sparse_original)

        real_CND = 0
        io = open("../grass/" * graph_name * "/" * string(CND_list[i]) * "/final_CND.txt", "r")
        while !eof(io)
            line = readline(io)
            real_CND = parse(Float64, line)
        end
        close(io)

        push!(iter_num, [CND_list[i], real_CND, iter_orginal, iter_updated])
    end

    return iter_num
end


function check_CND_with_K_itertion_exp(graph_name, CND;
    initial = false,
    exp = false,
    setup = (start_v= 10, interval = 10, end_v= 100)
    )

    CND_list = []
    density_list = []
    output_list = []

    # use with new excutable: incremental_update_new
    original_dense = "../grass/" * graph_name * "/" * string(CND) * "/Gmat_adj.mtx"
    original_sparse = "../grass/" * graph_name * "/" * string(CND) * "/Pmat_adj.mtx"
    input_sparse = "../grass/" * graph_name * "/" * string(CND) * "/Pmat_adj_0.mtx"
    updated_sparse = "../grass/" * graph_name * "/" * string(CND) * "/updated_adj_sparse.mtx"
    added_edges_path = "../grass/" * graph_name * "/" * string(CND) * "/ext.mtx"

    dense_edges, dense_weights = readMtx(original_dense, base = 1, type = "adj", weighted = true)
    sparse_edges, sparse_weights = readMtx(original_sparse, base = 1, type = "adj", weighted = true)
    added_edges, added_weights = readMtx(added_edges_path, base = 1, type = "adj", weighted = true)

    L_dense_original = create_laplacian(dense_edges, dense_weights)
    L_sparse_original = create_laplacian(sparse_edges, sparse_weights)

    number_of_vertices = V_size(dense_edges)
    println("Number of vertices = ", number_of_vertices)
    density = check_density(sparse_edges, number_of_vertices)
    println("Original sparsifier density = ", density)
    if initial # original CND and density
        CND = eigs(L_dense_original, L_sparse_original)
        println("Original sparsifier CND = ", maximum(real(CND[1])))
        append!(CND_list, maximum(real(CND[1])))
        append!(density_list, density)
        append!(output_list, "Original CND")
    end

    append!(dense_edges, added_edges)
    append!(dense_weights, added_weights)

    L_dense_updated = create_laplacian(dense_edges, dense_weights)


    if exp
        for K = setup[1]:setup[2]:setup[3]
            cmd = `./incremental_update_new`
            arg1 = input_sparse
            arg2 = added_edges_path
            arg3 = updated_sparse
            arg4 = K
            arg5 = 512 # walker
            cmd = `$cmd $arg1 $arg2 $arg3 $arg4 $arg5`
            output = read(cmd, String)
            println("Output: ", output)
            updated_edges, updated_weights = readMtx(updated_sparse, base=1, type="adj", weighted=true)
            L_sparse_updated = create_laplacian(updated_edges, updated_weights)
            CND = eigs(L_dense_updated, L_sparse_updated)
            append!(CND_list, maximum(real(CND[1])))
            append!(density_list, check_density(updated_edges, number_of_vertices))
            push!(output_list, output)
        end

    end

    return CND_list, density_list, output_list
end

function single_graph_iter_exp(graph_name, CND)

    added_edges = "../dataset/" * graph_name * "/ext.mtx"
    added_edges, added_weights = readMtx(added_edges, base=1, type="adj", weighted=true)

    original_dense = "../grass/" * graph_name * "/" * string(CND) * "/Gmat_adj.mtx"
    dense_edges, dense_weights = readMtx(original_dense, base = 1, type = "adj", weighted = true)

    original_sparse = "../grass/" * graph_name * "/" * string(CND) * "/Pmat_adj.mtx"
    sparse_edges, sparse_weights = readMtx(original_sparse, base = 1, type = "adj", weighted = true)
    L_dense_original = create_laplacian(dense_edges, dense_weights)

    original_density = check_density(sparse_edges, V_size(dense_edges))

    updated_sparse = "../grass/" * graph_name * "/" * string(CND) * "/updated_adj_sparse.mtx"
    updated_edges, updated_weights = readMtx(updated_sparse, base=1, type="adj", weighted=true)

    append!(dense_edges, added_edges)
    append!(dense_weights, added_weights)

    L_dense_updated = create_laplacian(dense_edges, dense_weights)
    L_sparse_updated = create_laplacian(updated_edges, updated_weights)

    updated_density = check_density(updated_edges, V_size(dense_edges))
    println("Original density: ", original_density)
    println("Updated density: ", updated_density)
    iter_updated = pcg_test(L_dense_updated, L_sparse_updated)

    return iter_updated

end

function test_checkBase()
    """
    Test the checkBase function with sample data
    """
    println("Testing checkBase function...")
    
    # Test with existing dataset files if available
    dataset_dirs = ["../dataset"]
    for dir in dataset_dirs
        if isdir(dir)
            for subdir in readdir(dir; join=true)
                if isdir(subdir)
                    # Look for common graph file patterns
                    for filename in ["dense.mtx", "adj_sparse.mtx", "ext.mtx"]
                        filepath = joinpath(subdir, filename)
                        if isfile(filepath)
                            base = checkBase(filepath)
                            println("File: $filepath -> Base: $base")
                        end
                    end
                end
            end
        end
    end
end

function test()
    println("Testing dyGRASS.jl module...")
    test_checkBase()
end
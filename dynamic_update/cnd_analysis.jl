include("../dyGRASS.jl")
using Arpack

dense_edges, dense_weight = readMtx("./adj_dense.mtx"; base = 1, type = "adj", weighted = true, sort = true)
sparse_edges, sparse_weight = readMtx("./adj_sparse.mtx"; base = 1, type = "adj", weighted = true, sort = true)


Lap_dense = create_laplacian(dense_edges, dense_weight)
Lap_sparse = create_laplacian(sparse_edges, sparse_weight)

CND = eigs(Lap_dense, Lap_sparse)

CND_1 = maximum(real(CND[1]))

println("CND: ", CND_1)

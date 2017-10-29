__precompile__()

module CuArrays

using CUDAdrv, CUDAnative
import CUDAnative: cudaconvert

export CuArray, CuVector, CuMatrix, cu

include("memory.jl")
include("array.jl")
include("utils.jl")
include("indexing.jl")
include("broadcast.jl")
include("reduction.jl")

include("../deps/ext.jl")
include("blas/BLAS.jl")
if libcusolver ≠ nothing
  include("cusolver/CuSolver.jl")
end
if libcudnn ≠ nothing
  include("cudnn/CUDNN.jl")
end
include("gpuarray_interface.jl")

end # module

module CuSolver

import Base: one, zero
using CUDAdrv
using ..CuArrays: CuArray, CuVector, CuMatrix, CuVecOrMat, libcusolver
using ..CuArrays.BLAS: cublasfill, cublasop, cublasside, cublasFillMode_t, cublasOperation_t, cublasSideMode_t

const BlasChar = Char

include("libcusolver_types.jl")

function statusmessage(status)
    if status == CUSOLVER_STATUS_SUCCESS
        return "cusolver success"
    elseif status == CUSOLVER_STATUS_NOT_INITIALIZED
        return "cusolver not initialized"
    elseif status == CUSOLVER_STATUS_ALLOC_FAILED
        return "cusolver allocation failed"
    elseif status == CUSOLVER_STATUS_INVALID_VALUE
        return "cusolver invalid value"
    elseif status == CUSOLVER_STATUS_ARCH_MISMATCH
        return "cusolver architecture mismatch"
    elseif status == CUSOLVER_STATUS_EXECUTION_FAILED
        return "cusolver execution failed"
    elseif status == CUSOLVER_STATUS_INTERNAL_ERROR
        return "cusolver internal error"
    elseif status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        return "cusolver matrix type not supported"
    else
        return "cusolver unknown status"
    end
end

# error handling function
function statuscheck(status)
    if status == CUSOLVER_STATUS_SUCCESS
        return nothing
    end
    warn("CUSOLVER error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    println()
    throw(statusmessage(status))
end

# Typedef needed by libcusolver
const cudaStream_t = Ptr{Void}

include("libcusolver.jl")

#setup cusolver handles
const cusolverSphandle = cusolverSpHandle_t[0]
const cusolverDnhandle = cusolverDnHandle_t[0]

function cusolverDestroy()
    cusolverSpDestroy(cusolverSphandle[1])
    cusolverDnDestroy(cusolverDnhandle[1])
end

function __init__()
    cusolverSpCreate(cusolverSphandle)
    cusolverDnCreate(cusolverDnhandle)
    #clean up handles at exit
    atexit(()->cusolverDestroy())
end

include("sparse.jl")
include("dense.jl")

end

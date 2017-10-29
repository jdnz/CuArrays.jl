import CUSPARSE: cusparseMatDescr_t, CudaSparseMatrixCSR, SparseChar, cusparseindex
import CUSPARSE: CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT

#enum cusolverStatus_t
#error messages from CUSOLVER

const cusolverStatus_t = UInt32
const CUSOLVER_STATUS_SUCCESS                   = 0
const CUSOLVER_STATUS_NOT_INITIALIZED           = 1
const CUSOLVER_STATUS_ALLOC_FAILED              = 2
const CUSOLVER_STATUS_INVALID_VALUE             = 3
const CUSOLVER_STATUS_ARCH_MISMATCH             = 4
const CUSOLVER_STATUS_EXECUTION_FAILED          = 5
const CUSOLVER_STATUS_INTERNAL_ERROR            = 6
const CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 7

const csrqrInfo_t = Ptr{Void}

# refactorization types

const cusolverRfNumericBoostReport_t = UInt32
const CUSOLVER_NUMERIC_BOOST_NOT_USED           = 0
const CUSOLVER_NUMERIC_BOOST_USED               = 1

const cusolverRfResetValuesFastMode_t = UInt32
const CUSOLVER_RESET_VALUES_FAST_MODE_OFF       = 0
const CUSOLVER_RESET_VALUES_FAST_MODE_ON        = 1

const cusolverRfFactorization_t = UInt32
const CUSOLVER_FACTORIZATION_ALG0               = 0
const CUSOLVER_FACTORIZATION_ALG1               = 1
const CUSOLVER_FACTORIZATION_ALG2               = 2

const cusolverRfTriangularSolve_t = UInt32
const CUSOLVER_TRIANGULAR_SOLVE_ALG0            = 0
const CUSOLVER_TRIANGULAR_SOLVE_ALG1            = 1
const CUSOLVER_TRIANGULAR_SOLVE_ALG2            = 2
const CUSOLVER_TRIANGULAR_SOLVE_ALG3            = 3

const cusolverRfUnitDiagonal_t = UInt32
const CUSOLVER_UNIT_DIAGONAL_STORED_L           = 0
const CUSOLVER_UNIT_DIAGONAL_STORED_U           = 1
const CUSOLVER_UNIT_DIAGONAL_ASSUMED_L          = 2
const CUSOLVER_UNIT_DIAGONAL_ASSUMED_U          = 3

const cusolverDnContext = Void
const cusolverDnHandle_t = Ptr{cusolverDnContext}
const cusolverSpContext = Void
const cusolverSpHandle_t = Ptr{cusolverSpContext}
const cusolverRfContext = Void
const cusolverRfHandle_t = Ptr{cusolverRfContext}

#complex numbers

const cuComplex = Complex{Float32}
const cuDoubleComplex = Complex{Float64}

const CusolverFloat = Union{Float64,Float32,Complex128,Complex64}
const CusolverReal = Union{Float64,Float32}
const CusolverComplex = Union{Complex128,Complex64}

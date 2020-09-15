#ifndef CUDA_ERROR_HANDLING
#define CUDA_ERROR_HANDLING

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>

// back compatibility start
static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
// back compatibility end

static void check_cuda_error(cudaError_t status, const char* file, int line)
{
    if (cudaSuccess != status) {
        printf("CUDA API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n", file, line, cudaGetErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

static void check_cusparse_error(cusparseStatus_t status, const char* file, int line)
{
    if (CUSPARSE_STATUS_SUCCESS != status) {
        printf("\nCUSPARSE status reference:\n");
        printf("CUSPARSE_STATUS_SUCCESS                   -> %d\n", CUSPARSE_STATUS_SUCCESS);
        printf("CUSPARSE_STATUS_NOT_INITIALIZED           -> %d\n", CUSPARSE_STATUS_NOT_INITIALIZED);
        printf("CUSPARSE_STATUS_ALLOC_FAILED              -> %d\n", CUSPARSE_STATUS_ALLOC_FAILED);
        printf("CUSPARSE_STATUS_INVALID_VALUE             -> %d\n", CUSPARSE_STATUS_INVALID_VALUE);
        printf("CUSPARSE_STATUS_ARCH_MISMATCH             -> %d\n", CUSPARSE_STATUS_ARCH_MISMATCH);
        printf("CUSPARSE_STATUS_EXECUTION_FAILED          -> %d\n", CUSPARSE_STATUS_EXECUTION_FAILED);
        printf("CUSPARSE_STATUS_INTERNAL_ERROR            -> %d\n", CUSPARSE_STATUS_INTERNAL_ERROR);
        printf("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED -> %d\n", CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        printf("CUSPARSE_STATUS_NOT_SUPPORTED             -> %d\n", CUSPARSE_STATUS_NOT_SUPPORTED);
        printf("CUSPARSE_STATUS_INSUFFICIENT_RESOURCES    -> %d\n", CUSPARSE_STATUS_INSUFFICIENT_RESOURCES);
        printf("\n");
        printf("CUSPARSE API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n", file, line, cusparseGetErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(err) (check_cuda_error(err, __FILE__, __LINE__))
#define CHECK_CUSPARSE(err) (check_cusparse_error(err, __FILE__, __LINE__))

#endif

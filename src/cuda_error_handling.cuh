#ifndef CUDA_ERROR_HANDLING
#define CUDA_ERROR_HANDLING

#include <cuda_runtime.h>
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

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (cudaSuccess != status) {                                                                                  \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }

#define CHECK_CUSPARSE(func)                                                                                                  \
    {                                                                                                                         \
        cusparseStatus_t status = (func);                                                                                     \
        if (CUSPARSE_STATUS_SUCCESS != status) {                                                                              \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                               \
        }                                                                                                                     \
    }

#endif

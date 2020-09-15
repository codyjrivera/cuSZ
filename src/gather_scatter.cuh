// 20-09-10

#ifndef GATHER_SCATTER
#define GATHER_SCATTER

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cuSZ {
namespace impl {

template <typename DType>
void new_gather(
    DType*    d_A,  //
    size_t    len,
    const int m,
    int*      nnz,
    int**     csrRowPtr,
    int**     csrColInd,
    DType**   csrVal);

template <typename DType>
void new_scatter(
    DType*    d_A,  //
    size_t    len,
    const int m,
    int*      nnz,
    int**     csrRowPtr,
    int**     csrColInd,
    DType**   csrVal);

void GatherOutlierUsingCusparse(
    float*  d_A,  //
    size_t  len,
    int&    nnzC,
    int**   csrRowPtrC,
    int**   csrColIndC,
    float** csrValC);

void GatherOutlierUsingCusparse(
    float*    d_A,  //
    size_t    len,
    const int m,  // m == n, and m is lda
    int&      nnzC,
    int**     csrRowPtrC,
    int**     csrColIndC,
    float**   csrValC);

}  // namespace impl
}  // namespace cuSZ

#endif
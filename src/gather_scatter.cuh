// 20-09-10

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cuSZ {
namespace impl {

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
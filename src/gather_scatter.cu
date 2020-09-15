// 20-09-10

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cassert>
#include <iostream>
using std::cout;
using std::endl;

#include "format.hh"
#include "gather_scatter.cuh"

template <typename DType>
void cuSZ::impl::new_gather(
    DType*    d_A,  //
    size_t    len,
    const int m,
    int*      nnz,
    int**     csrRowPtr,
    int**     csrColInd,
    DType**   csrVal)
{
    cusparseHandle_t   handle      = nullptr;
    cudaStream_t       stream      = nullptr;
    cusparseMatDescr_t descr       = nullptr;
    cusparseStatus_t   status      = CUSPARSE_STATUS_SUCCESS;
    cudaError_t        cudaStat1   = cudaSuccess;
    cudaError_t        cudaStat2   = cudaSuccess;
    cudaError_t        cudaStat3   = cudaSuccess;
    const int          lda         = m;
    const int          n           = m;
    int*               d_csrRowPtr = nullptr;
    int*               d_csrColInd = nullptr;
    DType*             d_csrVal    = nullptr;

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);  // 1. create stream
    assert(cudaSuccess == cudaStat1);                                       //
    status = cusparseCreate(&handle);                                       // 2. create handle
    assert(CUSPARSE_STATUS_SUCCESS == status);                              //
    status = cusparseSetStream(handle, stream);                             // 3. bind stream
    assert(CUSPARSE_STATUS_SUCCESS == status);                              //
    status = cusparseCreateMatDescr(&descr);                                // 4. create descr
    assert(CUSPARSE_STATUS_SUCCESS == status);                              //
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);               // zero based
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);                // type

    // query workspace
    // clang-format off
    cudaStat1 = cudaMalloc((void**)&d_csrRowPtr, sizeof(int)   * (m + 1));
    cudaStat2 = cudaMalloc((void**)&d_csrColInd, sizeof(int)   * *nnz   );
    cudaStat3 = cudaMalloc((void**)&d_csrVal,    sizeof(DType) * *nnz   );
    // clang-format on
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    // compute nnz
    int* d_nnzPerRow = nullptr;
    status           = cusparseSnnz(
        handle, CUSPARSE_DIRECTION_ROW,  // parsed by row
        m, n, descr, d_A, lda,           // descrption of d_A
        d_nnzPerRow, nnz);               // output
    assert(CUSPARSE_STATUS_SUCCESS == status);

    // step 5: dense to csr
    status = cusparseSdense2csr(
        handle,                               //
        m, n, descr, d_A, lda,                // descritpion of d_A
        d_nnzPerRow,                          // prefileld by nnz() func
        d_csrVal, d_csrRowPtr, d_csrColInd);  // output
    assert(CUSPARSE_STATUS_SUCCESS == status);

    // clang-format off
    cudaStat1 = cudaMemcpy(*csrRowPtr, d_csrRowPtr, sizeof(int)   * (m + 1), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(*csrColInd, d_csrColInd, sizeof(int)   * *nnz,    cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(*csrVal,    d_csrVal,    sizeof(DType) * *nnz,    cudaMemcpyDeviceToHost);
    // clang-format on
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    // clean up
    if (d_csrRowPtr) cudaFree(d_csrRowPtr);
    if (d_csrColInd) cudaFree(d_csrColInd);
    if (d_csrVal) cudaFree(d_csrVal);
    if (d_nnzPerRow) cudaFree(d_nnzPerRow);

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (descr) cusparseDestroyMatDescr(descr);
}

template void cuSZ::impl::new_gather<float>(float*, size_t, const int, int*, int**, int**, float**);

template <typename DType>
void cuSZ::impl::new_scatter(
    DType*    d_A,  //
    size_t    len,
    const int m,
    int*      nnz,
    int**     csrRowPtr,
    int**     csrColInd,
    DType**   csrVal)
{
    cusparseHandle_t   handle      = nullptr;
    cudaStream_t       stream      = nullptr;
    cusparseMatDescr_t descr       = nullptr;
    cusparseStatus_t   status      = CUSPARSE_STATUS_SUCCESS;
    cudaError_t        cudaStat1   = cudaSuccess;
    cudaError_t        cudaStat2   = cudaSuccess;
    cudaError_t        cudaStat3   = cudaSuccess;
    const int          lda         = m;
    const int          n           = m;
    int*               d_csrRowPtr = nullptr;
    int*               d_csrColInd = nullptr;
    DType*             d_csrVal    = nullptr;

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);  // 1. create stream
    assert(cudaSuccess == cudaStat1);                                       //
    status = cusparseCreate(&handle);                                       // 2. create handle
    assert(CUSPARSE_STATUS_SUCCESS == status);                              //
    status = cusparseSetStream(handle, stream);                             // 3. bind stream
    assert(CUSPARSE_STATUS_SUCCESS == status);                              //
    status = cusparseCreateMatDescr(&descr);                                // 4. create descr
    assert(CUSPARSE_STATUS_SUCCESS == status);                              //
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);               //
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);                //

    // set space
    // clang-format off
    cudaStat1 = cudaMemcpy(d_csrRowPtr, *csrRowPtr, sizeof(int)   * (m + 1), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_csrColInd, *csrColInd, sizeof(int)   * *nnz,    cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrVal,    *csrVal,    sizeof(DType) * *nnz,    cudaMemcpyHostToDevice);
    // clang-format on
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    // fill
    status = cusparseScsr2dense(handle, m, n, descr, d_csrVal, d_csrRowPtr, d_csrColInd, d_A, lda);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    if (d_csrRowPtr) cudaFree(d_csrRowPtr);
    if (d_csrColInd) cudaFree(d_csrColInd);
    if (d_csrVal) cudaFree(d_csrVal);

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (descr) cusparseDestroyMatDescr(descr);
}

template void cuSZ::impl::new_scatter<float>(float*, size_t, const int, int*, int**, int**, float**);

void cuSZ::impl::GatherOutlierUsingCusparse(
    float*    d_A,  //
    size_t    len,
    const int m,
    int&      nnzC,
    int**     csrRowPtrC,
    int**     csrColIndC,
    float**   csrValC)
{
    cusparseHandle_t   handle    = nullptr;
    cudaStream_t       stream    = nullptr;
    cusparseMatDescr_t descrC    = nullptr;
    cusparseStatus_t   status    = CUSPARSE_STATUS_SUCCESS;
    cudaError_t        cudaStat1 = cudaSuccess;
    cudaError_t        cudaStat2 = cudaSuccess;
    cudaError_t        cudaStat3 = cudaSuccess;
    // cudaError_t cudaStat4 = cudaSuccess;
    // cudaError_t cudaStat5 = cudaSuccess;
    // const int m           = 1;
    // const int n           = len;
    const int lda = m;
    const int n   = m;  // square

    // int*   csrRowPtrC = nullptr;
    // int*   csrColIndC = nullptr;
    // float* csrValC    = nullptr;
    // float* d_A        = nullptr;
    int*   d_csrRowPtrC = nullptr;
    int*   d_csrColIndC = nullptr;
    float* d_csrValC    = nullptr;

    size_t lworkInBytes = 0;
    char*  d_work       = nullptr;

    //    int nnzC = 0;

    float threshold = 0; /* remove Aij <= 4.1 */

    /* step 1: create cusparse handle, bind a stream */
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    /* step 2: configuration of matrix C */
    status = cusparseCreateMatDescr(&descrC);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);

    //    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) * lda * n);
    cudaStat2 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int) * (m + 1));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query workspace */
    //    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
    //    assert(cudaSuccess == cudaStat1);

    status = cusparseSpruneDense2csr_bufferSizeExt(  //
        handle,                                      //
        m,                                           //
        n,                                           //
        d_A,                                         //
        lda,                                         //
        &threshold,                                  //
        descrC,                                      //
        d_csrValC,                                   //
        d_csrRowPtrC,                                //
        d_csrColIndC,                                //
        &lworkInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    //    printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);

    if (nullptr != d_work) {
        cudaFree(d_work);
    }
    cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
    assert(cudaSuccess == cudaStat1);

    /* step 4: compute csrRowPtrC and nnzC */
    status = cusparseSpruneDense2csrNnz(  //
        handle,                           //
        m,                                //
        n,                                //
        d_A,                              //
        lda,                              //
        &threshold,                       //
        descrC,                           //
        d_csrRowPtrC,                     //
        &nnzC,                            // host
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    if (0 == nnzC) cout << log_info << "No outlier." << endl;

    /* step 5: compute csrColIndC and csrValC */
    cudaStat1 = cudaMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC);
    cudaStat2 = cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    status = cusparseSpruneDense2csr(  //
        handle,                        //
        m,                             //
        n,                             //
        d_A,                           //
        lda,                           //
        &threshold,                    //
        descrC,                        //
        d_csrValC,                     //
        d_csrRowPtrC,                  //
        d_csrColIndC,                  //
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    /* step 6: output C */
    //    csrRowPtrC = (int*)malloc(sizeof(int) * (m + 1));
    //    csrColIndC = (int*)malloc(sizeof(int) * nnzC);
    //    csrValC    = (float*)malloc(sizeof(float) * nnzC);
    *csrRowPtrC = new int[m + 1];
    *csrColIndC = new int[nnzC];
    *csrValC    = new float[nnzC];
    assert(nullptr != csrRowPtrC);
    assert(nullptr != csrColIndC);
    assert(nullptr != csrValC);

    cudaStat1 = cudaMemcpy(*csrRowPtrC, d_csrRowPtrC, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(*csrColIndC, d_csrColIndC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(*csrValC, d_csrValC, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    //    printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");

    /* free resources */
    if (d_A) cudaFree(d_A);
    if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
    if (d_csrColIndC) cudaFree(d_csrColIndC);
    if (d_csrValC) cudaFree(d_csrValC);

    //    if (csrRowPtrC) free(csrRowPtrC);
    //    if (csrColIndC) free(csrColIndC);
    //    if (csrValC) free(csrValC);

    //    for (auto i = 0; i < 200; i++) cout << i << "\t" << csrColIndC[i] << "\t" << csrValC[i] << endl;

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (descrC) cusparseDestroyMatDescr(descrC);

    //    cudaDeviceReset();
}

void cuSZ::impl::GatherOutlierUsingCusparse(
    float*  d_A,  //
    size_t  len,
    int&    nnzC,
    int**   csrRowPtrC,
    int**   csrColIndC,
    float** csrValC)
{
    cusparseHandle_t   handle    = nullptr;
    cudaStream_t       stream    = nullptr;
    cusparseMatDescr_t descrC    = nullptr;
    cusparseStatus_t   status    = CUSPARSE_STATUS_SUCCESS;
    cudaError_t        cudaStat1 = cudaSuccess;
    cudaError_t        cudaStat2 = cudaSuccess;
    cudaError_t        cudaStat3 = cudaSuccess;
    //    cudaError_t        cudaStat4 = cudaSuccess;
    //    cudaError_t        cudaStat5 = cudaSuccess;
    const int m   = 1;
    const int n   = len;
    const int lda = m;

    //    int*   csrRowPtrC = nullptr;
    //    int*   csrColIndC = nullptr;
    //    float* csrValC    = nullptr;

    //    float* d_A          = nullptr;
    int*   d_csrRowPtrC = nullptr;
    int*   d_csrColIndC = nullptr;
    float* d_csrValC    = nullptr;

    size_t lworkInBytes = 0;
    char*  d_work       = nullptr;

    //    int nnzC = 0;

    float threshold = 0; /* remove Aij <= 4.1 */

    /* step 1: create cusparse handle, bind a stream */
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    /* step 2: configuration of matrix C */
    status = cusparseCreateMatDescr(&descrC);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);

    //    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) * lda * n);
    cudaStat2 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int) * (m + 1));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query workspace */
    //    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
    //    assert(cudaSuccess == cudaStat1);

    status = cusparseSpruneDense2csr_bufferSizeExt(  //
        handle,                                      //
        m,                                           //
        n,                                           //
        d_A,                                         //
        lda,                                         //
        &threshold,                                  //
        descrC,                                      //
        d_csrValC,                                   //
        d_csrRowPtrC,                                //
        d_csrColIndC,                                //
        &lworkInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    //    printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);

    if (nullptr != d_work) {
        cudaFree(d_work);
    }
    cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
    assert(cudaSuccess == cudaStat1);

    /* step 4: compute csrRowPtrC and nnzC */
    status = cusparseSpruneDense2csrNnz(  //
        handle,                           //
        m,                                //
        n,                                //
        d_A,                              //
        lda,                              //
        &threshold,                       //
        descrC,                           //
        d_csrRowPtrC,                     //
        &nnzC,                            // host
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    if (0 == nnzC) cout << log_info << "No outlier." << endl;

    /* step 5: compute csrColIndC and csrValC */
    cudaStat1 = cudaMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC);
    cudaStat2 = cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    status = cusparseSpruneDense2csr(  //
        handle,                        //
        m,                             //
        n,                             //
        d_A,                           //
        lda,                           //
        &threshold,                    //
        descrC,                        //
        d_csrValC,                     //
        d_csrRowPtrC,                  //
        d_csrColIndC,                  //
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    /* step 6: output C */
    //    csrRowPtrC = (int*)malloc(sizeof(int) * (m + 1));
    //    csrColIndC = (int*)malloc(sizeof(int) * nnzC);
    //    csrValC    = (float*)malloc(sizeof(float) * nnzC);
    *csrRowPtrC = new int[m + 1];
    *csrColIndC = new int[nnzC];
    *csrValC    = new float[nnzC];
    assert(nullptr != csrRowPtrC);
    assert(nullptr != csrColIndC);
    assert(nullptr != csrValC);

    cudaStat1 = cudaMemcpy(*csrRowPtrC, d_csrRowPtrC, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(*csrColIndC, d_csrColIndC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(*csrValC, d_csrValC, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    //    printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");

    /* free resources */
    if (d_A) cudaFree(d_A);
    if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
    if (d_csrColIndC) cudaFree(d_csrColIndC);
    if (d_csrValC) cudaFree(d_csrValC);

    //    if (csrRowPtrC) free(csrRowPtrC);
    //    if (csrColIndC) free(csrColIndC);
    //    if (csrValC) free(csrValC);

    //    for (auto i = 0; i < 200; i++) cout << i << "\t" << csrColIndC[i] << "\t" << csrValC[i] << endl;

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (descrC) cusparseDestroyMatDescr(descrC);

    //    cudaDeviceReset();
}
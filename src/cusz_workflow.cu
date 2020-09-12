
#include <bits/stdint-uintn.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cxxabi.h>
#include <bitset>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "argparse.hh"
#include "constants.hh"
#include "cuda_error_handling.cuh"
#include "cuda_mem.cuh"
#include "cusz_dryrun.cuh"
#include "cusz_dualquant.cuh"
#include "cusz_workflow.cuh"
#include "filter.cuh"
#include "format.hh"
#include "gather_scatter.cuh"
#include "huffman_workflow.cuh"
#include "io.hh"
#include "verify.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

const int gpu_B_1d = 32;
const int gpu_B_2d = 16;
const int gpu_B_3d = 8;

// moved to const_device.cuh
__constant__ int    symb_dims[16];
__constant__ double symb_ebs[4];

template <typename T>
__global__ void CountOutlier(T const* const outlier, int* _d_n_outlier, size_t len)
{
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= len) return;
    if (outlier[gid] != 0) atomicAdd(_d_n_outlier, 1);
    __syncthreads();
}

//__global__ void CountOutlier<float>(float const* const outlier, int* _d_n_outlier, size_t len);
//__global__ void CountOutlier<double>(double const* const outlier, int* _d_n_outlier, size_t len);

template <typename T, typename Q>
void cuSZ::workflow::PdQ(T* d_data, Q* d_bcode, size_t* dims_L16, double* ebs_L4)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    auto  d_ebs_L4   = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs_L4, 4);
    void* args[]     = {&d_data, &d_bcode, &d_dims_L16, &d_ebs_L4};

    // testing constant memory
    auto dims_inttype = new int[16];
    for (auto i = 0; i < 16; i++) dims_inttype[i] = dims_L16[i];
    cudaMemcpyToSymbol(symb_dims, dims_inttype, 16 * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(symb_ebs, ebs_L4, 4 * sizeof(double), 0, cudaMemcpyHostToDevice);
    // void* args2[] = {&d_data, &d_bcode}; unreferenced

    if (dims_L16[nDIM] == 1) {
        dim3 blockNum(dims_L16[nBLK0]);
        dim3 threadNum(gpu_B_1d);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_1d1l<T, Q, gpu_B_1d>,  //
            blockNum, threadNum, args, 0, nullptr);
        /*
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_1d1l<T, Q, gpu_B_1d>,  //
            blockNum, threadNum, args2, gpu_B_1d * sizeof(T), nullptr);
        */
    }
    else if (dims_L16[nDIM] == 2) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1]);
        dim3 threadNum(gpu_B_2d, gpu_B_2d);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_2d1l<T, Q, gpu_B_2d>,  //
            blockNum, threadNum, args, (gpu_B_2d + 1) * (gpu_B_2d + 1) * sizeof(T), nullptr);
        /*
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_2d1l<T, Q, gpu_B_2d>,  //
            blockNum, threadNum, args2, (gpu_B_2d) * (gpu_B_2d) * sizeof(T), nullptr);
        */
    }
    else if (dims_L16[nDIM] == 3) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1], dims_L16[nBLK2]);
        dim3 threadNum(gpu_B_3d, gpu_B_3d, gpu_B_3d);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_3d1l<T, Q, gpu_B_3d>,  //
            blockNum, threadNum, args, (gpu_B_3d + 1) * (gpu_B_3d + 1) * (gpu_B_3d + 1) * sizeof(T), nullptr);
        /*
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_3d1l_new<T, Q, gpu_B_3d>,  //
            blockNum, threadNum, args2, (gpu_B_3d + 1) * (gpu_B_3d + 1) * (gpu_B_3d + 1) * sizeof(T), nullptr);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_3d1l<T, Q, gpu_B_3d>,  //
            blockNum, threadNum, args2, (gpu_B_3d) * (gpu_B_3d) * (gpu_B_3d) * sizeof(T), nullptr);
        */
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
}

template void cuSZ::workflow::PdQ<float, uint8_t>(float* d_data, uint8_t* d_bcode, size_t* dims_L16, double* ebs_L4);
template void cuSZ::workflow::PdQ<float, uint16_t>(float* d_data, uint16_t* d_bcode, size_t* dims_L16, double* ebs_L4);
template void cuSZ::workflow::PdQ<float, uint32_t>(float* d_data, uint32_t* d_bcode, size_t* dims_L16, double* ebs_L4);
// template void cuSZ::workflow::PdQ<double, uint8_t>(double* d_data, uint8_t* d_bcode, size_t* dims_L16, double* ebs_L4);
// template void cuSZ::workflow::PdQ<double, uint16_t>(double* d_data, uint16_t* d_bcode, size_t* dims_L16, double* ebs_L4);
// template void cuSZ::workflow::PdQ<double, uint32_t>(double* d_data, uint32_t* d_bcode, size_t* dims_L16, double* ebs_L4);

// struct KernelConfig {
//    dim3   gridDim;
//    dim3   blockDim;
//    size_t shmem_size;
//
//    KernelConfig(size_t* dims, size_t B)
//    {
//        switch (dims[nDIM]) {
//            case 1:
//                gridDim  = {dims[L16[nBLK0]]};
//                blockDim = {B};
//                break;
//            case 2:
//                gridDim  = {dims[nBLK0], dims[nBLK1]};
//                blockDim = {B};
//                break;
//            case 3:
//                gridDim  = {dims[L16[nBLK0]]};
//                blockDim = {B};
//                break;
//            default:
//                break;
//        }
//    }
//};

template <typename T, typename Q>
void cuSZ::workflow::ReversedPdQ(T* d_xdata, Q* d_bcode, T* d_outlier, size_t* dims_L16, double _2eb)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    void* args[]     = {&d_xdata, &d_outlier, &d_bcode, &d_dims_L16, &_2eb};

    if (dims_L16[nDIM] == 1) {
        const static size_t p = gpu_B_1d;

        dim3 thread_num(p);
        dim3 block_num((dims_L16[nBLK0] - 1) / p + 1);
        cudaLaunchKernel((void*)PdQ::x_lorenzo_1d1l<T, Q, gpu_B_1d>, block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 2) {
        const static size_t p = gpu_B_2d;

        dim3 thread_num(p, p);
        dim3 block_num(
            (dims_L16[nBLK0] - 1) / p + 1,   //
            (dims_L16[nBLK1] - 1) / p + 1);  //
        cudaLaunchKernel((void*)PdQ::x_lorenzo_2d1l<T, Q, gpu_B_2d>, block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 3) {
        const static size_t p = gpu_B_3d;

        dim3 thread_num(p, p, p);
        dim3 block_num(
            (dims_L16[nBLK0] - 1) / p + 1,   //
            (dims_L16[nBLK1] - 1) / p + 1,   //
            (dims_L16[nBLK2] - 1) / p + 1);  //
        cudaLaunchKernel((void*)PdQ::x_lorenzo_3d1l<T, Q, gpu_B_3d>, block_num, thread_num, args, 0, nullptr);
        // PdQ::x_lorenzo_3d1l<T, Q, gpu_B_3d><<<block_num, thread_num>>>(d_xdata, d_outlier, d_bcode, d_dims_L16, _2eb);
    }
    else {
        cerr << log_err << "no 4D" << endl;
    }
    cudaDeviceSynchronize();

    cudaFree(d_dims_L16);
}

// template <typename T>
// __global__ void cuSZ::workflow::Condenser(T* outlier, int* meta, size_t BLK, size_t nBLK)
// {
//     auto id = blockDim.x * blockIdx.x + threadIdx.x;
//     if (id >= nBLK) return;
//     int count = 0;
//     for (auto i = 0; i < BLK; i++)
//         if (outlier[i] != 0) outlier[count++] = outlier[i];

//     meta[id] = count;
// }

// void cuSZ::workflow::DeflateOutlierUsingCuSparse(
//     float*  d_A,  //
//     size_t  len,
//     int&    nnzC,
//     int**   csrRowPtrC,
//     int**   csrColIndC,
//     float** csrValC)
// {
//     cusparseHandle_t   handle    = nullptr;
//     cudaStream_t       stream    = nullptr;
//     cusparseMatDescr_t descrC    = nullptr;
//     cusparseStatus_t   status    = CUSPARSE_STATUS_SUCCESS;
//     cudaError_t        cudaStat1 = cudaSuccess;
//     cudaError_t        cudaStat2 = cudaSuccess;
//     cudaError_t        cudaStat3 = cudaSuccess;
//     //    cudaError_t        cudaStat4 = cudaSuccess;
//     //    cudaError_t        cudaStat5 = cudaSuccess;
//     const int m   = 1;
//     const int n   = len;
//     const int lda = m;

//     //    int*   csrRowPtrC = nullptr;
//     //    int*   csrColIndC = nullptr;
//     //    float* csrValC    = nullptr;

//     //    float* d_A          = nullptr;
//     int*   d_csrRowPtrC = nullptr;
//     int*   d_csrColIndC = nullptr;
//     float* d_csrValC    = nullptr;

//     size_t lworkInBytes = 0;
//     char*  d_work       = nullptr;

//     //    int nnzC = 0;

//     float threshold = 0; /* remove Aij <= 4.1 */

//     /* step 1: create cusparse handle, bind a stream */
//     cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
//     assert(cudaSuccess == cudaStat1);

//     status = cusparseCreate(&handle);
//     assert(CUSPARSE_STATUS_SUCCESS == status);

//     status = cusparseSetStream(handle, stream);
//     assert(CUSPARSE_STATUS_SUCCESS == status);

//     /* step 2: configuration of matrix C */
//     status = cusparseCreateMatDescr(&descrC);
//     assert(CUSPARSE_STATUS_SUCCESS == status);

//     cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
//     cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);

//     //    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) * lda * n);
//     cudaStat2 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int) * (m + 1));
//     assert(cudaSuccess == cudaStat1);
//     assert(cudaSuccess == cudaStat2);

//     /* step 3: query workspace */
//     //    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
//     //    assert(cudaSuccess == cudaStat1);

//     status = cusparseSpruneDense2csr_bufferSizeExt(  //
//         handle,                                      //
//         m,                                           //
//         n,                                           //
//         d_A,                                         //
//         lda,                                         //
//         &threshold,                                  //
//         descrC,                                      //
//         d_csrValC,                                   //
//         d_csrRowPtrC,                                //
//         d_csrColIndC,                                //
//         &lworkInBytes);
//     assert(CUSPARSE_STATUS_SUCCESS == status);

//     //    printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);

//     if (nullptr != d_work) {
//         cudaFree(d_work);
//     }
//     cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
//     assert(cudaSuccess == cudaStat1);

//     /* step 4: compute csrRowPtrC and nnzC */
//     status = cusparseSpruneDense2csrNnz(  //
//         handle,                           //
//         m,                                //
//         n,                                //
//         d_A,                              //
//         lda,                              //
//         &threshold,                       //
//         descrC,                           //
//         d_csrRowPtrC,                     //
//         &nnzC,                            // host
//         d_work);
//     assert(CUSPARSE_STATUS_SUCCESS == status);
//     cudaStat1 = cudaDeviceSynchronize();
//     assert(cudaSuccess == cudaStat1);

//     if (0 == nnzC) cout << log_info << "No outlier." << endl;

//     /* step 5: compute csrColIndC and csrValC */
//     cudaStat1 = cudaMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC);
//     cudaStat2 = cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
//     assert(cudaSuccess == cudaStat1);
//     assert(cudaSuccess == cudaStat2);

//     status = cusparseSpruneDense2csr(  //
//         handle,                        //
//         m,                             //
//         n,                             //
//         d_A,                           //
//         lda,                           //
//         &threshold,                    //
//         descrC,                        //
//         d_csrValC,                     //
//         d_csrRowPtrC,                  //
//         d_csrColIndC,                  //
//         d_work);
//     assert(CUSPARSE_STATUS_SUCCESS == status);
//     cudaStat1 = cudaDeviceSynchronize();
//     assert(cudaSuccess == cudaStat1);

//     /* step 6: output C */
//     //    csrRowPtrC = (int*)malloc(sizeof(int) * (m + 1));
//     //    csrColIndC = (int*)malloc(sizeof(int) * nnzC);
//     //    csrValC    = (float*)malloc(sizeof(float) * nnzC);
//     *csrRowPtrC = new int[m + 1];
//     *csrColIndC = new int[nnzC];
//     *csrValC    = new float[nnzC];
//     assert(nullptr != csrRowPtrC);
//     assert(nullptr != csrColIndC);
//     assert(nullptr != csrValC);

//     cudaStat1 = cudaMemcpy(*csrRowPtrC, d_csrRowPtrC, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
//     cudaStat2 = cudaMemcpy(*csrColIndC, d_csrColIndC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
//     cudaStat3 = cudaMemcpy(*csrValC, d_csrValC, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);
//     assert(cudaSuccess == cudaStat1);
//     assert(cudaSuccess == cudaStat2);
//     assert(cudaSuccess == cudaStat3);

//     //    printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");

//     /* free resources */
//     if (d_A) cudaFree(d_A);
//     if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
//     if (d_csrColIndC) cudaFree(d_csrColIndC);
//     if (d_csrValC) cudaFree(d_csrValC);

//     //    if (csrRowPtrC) free(csrRowPtrC);
//     //    if (csrColIndC) free(csrColIndC);
//     //    if (csrValC) free(csrValC);

//     //    for (auto i = 0; i < 200; i++) cout << i << "\t" << csrColIndC[i] << "\t" << csrValC[i] << endl;

//     if (handle) cusparseDestroy(handle);
//     if (stream) cudaStreamDestroy(stream);
//     if (descrC) cusparseDestroyMatDescr(descrC);

//     //    cudaDeviceReset();
// }

// template <typename T>
// size_t* cuSZ::workflow::DeflateOutlier(T* d_outlier, T* outlier, int* meta, size_t len, size_t BLK, size_t nBLK, int blockDim)
// {
//     // get to know num of non-zeros
//     int* d_nnz;
//     int  nnz = 0;
//     cudaMalloc(&d_nnz, sizeof(int));
//     cudaMemset(d_nnz, 0, sizeof(int));
//     CountOutlier<<<(len - 1) / 256 + 1, 256>>>(d_outlier, d_nnz, len);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost);
//     // deflate
//     meta = new int[nBLK]();
//     int* d_meta;
//     cudaMalloc(&d_meta, nBLK * sizeof(int));
//     cudaMemset(d_meta, 0, nBLK = sizeof(int));
//     cudaDeviceSynchronize();
//     // copy back to host
//     outlier = new T[nnz]();

//     cudaMemcpy(meta, d_meta, nBLK * sizeof(int), cudaMemcpyDeviceToHost);
//     for (auto i = 0, begin = 0; i < nBLK; i++) {
//         cudaMemcpy(outlier + begin, d_outlier + i * BLK, meta[i] * sizeof(T), cudaMemcpyDeviceToHost);
//         begin += meta[i];
//     }

//     return outlier;
// }

template <typename T, typename Q>
void cuSZ::workflow::VerifyHuffman(string const& fi, size_t len, Q* xbcode, int chunk_size, size_t* dims_L16, double* ebs_L4)
{
    // TODO error handling from invalid read
    cout << log_info << "Redo PdQ just to get quantization dump." << endl;

    auto veri_data    = io::ReadBinaryFile<T>(fi, len);
    T*   veri_d_data  = mem::CreateDeviceSpaceAndMemcpyFromHost(veri_data, len);
    auto veri_d_bcode = mem::CreateCUDASpace<Q>(len);
    PdQ(veri_d_data, veri_d_bcode, dims_L16, ebs_L4);

    auto veri_bcode = mem::CreateHostSpaceAndMemcpyFromDevice(veri_d_bcode, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xbcode[i] != veri_bcode[i]) count++;
    if (count != 0)
        cerr << log_err << "percentage of not being equal: " << count / (1.0 * len) << "\n";
    else
        cout << log_info << "Decoded correctly." << endl;

    if (count != 0) {
        //        auto chunk_size = ap->huffman_chunk;
        auto n_chunk = (len - 1) / chunk_size + 1;
        for (auto c = 0; c < n_chunk; c++) {
            auto chunk_id_printed   = false;
            auto prev_point_printed = false;
            for (auto i = 0; i < chunk_size; i++) {
                auto idx = i + c * chunk_size;
                if (idx >= len) break;
                if (xbcode[idx] != xbcode[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t";
                        cerr << "start@ " << c * chunk_size << "\tend@ " << (c + 1) * chunk_size - 1 << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * chunk_size) {  // not first point
                            cerr << "PREV-idx:" << idx - 1 << "\t" << xbcode[idx - 1] << "\t" << xbcode[idx - 1] << endl;
                        }
                        else {
                            cerr << "wrong at first point!" << endl;
                        }
                        prev_point_printed = true;
                    }
                    cerr << "idx:" << idx << "\tdecoded: " << xbcode[idx] << "\tori: " << xbcode[idx] << endl;
                }
            }
        }
    }

    cudaFree(veri_d_bcode);
    cudaFree(veri_d_data);
    delete[] veri_bcode;
    delete[] veri_data;
    // end of if count
}

template <typename T, typename Q, typename H>
void cuSZ::workflow::Compress(
    std::string& fi,
    size_t*      dims_L16,
    double*      ebs_L4,
    size_t&      nnz_outlier,
    size_t&      n_bits,
    size_t&      n_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap)
{
    string fo_bcode, fo_outlier, fo_outlier_new;

    string fo_cdata = fi + ".sza";
    int    bw       = sizeof(Q) * 8;
    fo_bcode        = fi + ".b" + std::to_string(bw);
    fo_outlier_new  = fi + ".b" + std::to_string(bw) + "outlier_new";

    // TODO to use a struct
    size_t len         = dims_L16[LEN];
    auto   padded_edge = GetEdgeOfReinterpretedSquare(len);
    auto   padded_len  = padded_edge * padded_edge;

    cout << log_info << "padded edge:\t" << padded_edge << "\tpadded_len:\t" << padded_len << endl;

    // old: use the orignal length as it is
    // auto data   = io::ReadBinaryFile<T>(fi, len);
    // T*   d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, len);
    // new: use padded length for outlier gather/scatter
    auto data = new T[padded_len]();
    io::ReadBinaryFile<T>(fi, data, len);
    T* d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, padded_len);

    //    for (auto i = 150; i < 200; i++) cout << data[i] << endl;

    if (ap->dry_run) {
        cout << "\n" << log_info << "Commencing dry-run..." << endl;
        DryRun(data, d_data, fi, dims_L16, ebs_L4);
        exit(0);
    }
    cout << "\n" << log_info << "Commencing compression..." << endl;

    auto d_bcode = mem::CreateCUDASpace<Q>(len);  // quant. code is not needed for dry-run

    // prediction-quantization
    PdQ(d_data, d_bcode, dims_L16, ebs_L4);

    // dealing with outlier
    int*   outlier_csrRowPtrC = nullptr;  //
    int*   outlier_csrColIndC = nullptr;  // column major, real index
    float* outlier_csrValC    = nullptr;  // outlier values; TODO template
    int    nnzC               = 0;

    // old, 1D
    // DeflateOutlierUsingCuSparse(d_data, len, nnzC, &outlier_csrRowPtrC, &outlier_csrColIndC, &outlier_csrValC);
    // new, reinterpreted 2D
    DeflateOutlierUsingCuSparse(d_data, padded_len, padded_edge, nnzC, &outlier_csrRowPtrC, &outlier_csrColIndC, &outlier_csrValC);

    nnz_outlier = nnzC;  // TODO temporarily nnzC is not archived because num_outlier is available out of this scope
    /*
    // old output of outlier as spm
    auto outlier_bin = new uint8_t[nnzC * (sizeof(int) + sizeof(float))];
    memcpy(outlier_bin, (uint8_t*)outlier_csrColIndC, nnzC * sizeof(int));
    memcpy(outlier_bin + nnzC * sizeof(int), (uint8_t*)outlier_csrValC, nnzC * sizeof(float));
    cout << log_info << "nnz/num.outlier:\t" << nnz_outlier << "\t(" << (nnz_outlier / 1.0 / len * 100) << "%)" << endl;
    // cout << log_info << "Dumping outlier..." << endl;
    io::WriteBinaryFile(outlier_bin, nnzC * (sizeof(int) + sizeof(float)), &fo_outlier_new);
    */
    // new output of outlier as spm
    // clang-format off
    auto bytelen_csrRowPtrC = sizeof(int)   * (padded_edge + 1);
    auto bytelen_csrColIndC = sizeof(int)   *  nnzC;
    auto bytelen_csrValC    = sizeof(float) *  nnzC;
    auto bytelen_total      = bytelen_csrRowPtrC + bytelen_csrColIndC + bytelen_csrValC;
    auto outlier_bin        = new uint8_t[bytelen_total];
    memcpy(outlier_bin,                                           (uint8_t*)outlier_csrColIndC, bytelen_csrRowPtrC);
    memcpy(outlier_bin + bytelen_csrRowPtrC,                      (uint8_t*)outlier_csrValC,    bytelen_csrColIndC);
    memcpy(outlier_bin + bytelen_csrRowPtrC + bytelen_csrColIndC, (uint8_t*)outlier_csrColIndC, bytelen_csrValC);
    // clang-format on

    cout << log_info << "outlier_bin byte length:\t" << bytelen_total << endl;
    cout << log_info << "nnz/num.outlier:\t" << nnz_outlier << "\t(" << (nnz_outlier / 1.0 / len * 100) << "%)" << endl;

    Q* bcode;
    if (ap->skip_huffman) {
        // cout << log_info << "Skipping Huffman..." << endl;
        bcode = mem::CreateHostSpaceAndMemcpyFromDevice(d_bcode, len);
        io::WriteBinaryFile(bcode, len, &fo_bcode);
        cout << log_info << "Compression finished, saved quant.code (Huffman skipped).\n" << endl;
        return;
    }
    typedef std::tuple<size_t, size_t, size_t> tuple3ul;

    // huffman encoding
    std::tuple<size_t, size_t, size_t> t = HuffmanEncode<Q, H>(fo_bcode, d_bcode, len, ap->huffman_chunk, dims_L16[CAP]);

    std::tie(n_bits, n_uInt, huffman_metadata_size) = t;

    cout << log_info << "Compression finished, saved Huffman encoded quant.code.\n" << endl;

    // clean up
    delete[] data;
    delete[] outlier_csrColIndC;
    delete[] outlier_csrValC;
    delete[] outlier_csrRowPtrC;
    delete[] outlier_bin;

    cudaFree(d_data);
}

template <typename T, typename Q, typename H>
void cuSZ::workflow::Decompress(
    std::string& fi,  //
    size_t*      dims_L16,
    double*      ebs_L4,
    size_t&      nnz_outlier,
    size_t&      total_bits,
    size_t&      total_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap)
{
    //    string f_archive = fi + ".sza"; // TODO
    string f_extract = ap->alt_xout_name.empty() ? fi + ".szx" : ap->alt_xout_name;
    string fi_bcode_base, fi_bcode_after_huffman, fi_outlier, fi_outlier_as_cuspm;

    fi_bcode_base       = fi + ".b" + std::to_string(sizeof(Q) * 8);
    fi_outlier_as_cuspm = fi_bcode_base + "outlier_new";
    //    fi_bcode_after_huffman = fi_bcode_base + ".x";

    auto dict_size = dims_L16[CAP];
    auto len       = dims_L16[LEN];

    // TODO to use a struct
    // add padding
    auto padded_edge = GetEdgeOfReinterpretedSquare(len);
    auto padded_len  = padded_edge * padded_edge;

    cout << log_info << "Commencing decompression..." << endl;

    Q* xbcode;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (ap->skip_huffman) {
        cout << log_info << "Huffman skipped, reading quant. code from filesystem..." << endl;
        xbcode = io::ReadBinaryFile<Q>(fi_bcode_base, len);
    }
    else {
        cout << log_info << "Getting quant. code from Huffman decoding..." << endl;
        xbcode = HuffmanDecode<Q, H>(fi_bcode_base, len, ap->huffman_chunk, total_uInt, dict_size);
        if (ap->verify_huffman) {
            cout << log_info << "Verifying Huffman codec..." << endl;
            VerifyHuffman<T, Q>(fi, len, xbcode, ap->huffman_chunk, dims_L16, ebs_L4);
        }
    }
    auto d_bcode = mem::CreateDeviceSpaceAndMemcpyFromHost(xbcode, len);

    /*
    // #ifdef OLD_OUTLIER_METHOD
    //     auto outlier   = io::ReadBinaryFile<T>(fi_outlier, len);
    //     auto d_outlier = mem::CreateDeviceSpaceAndMemcpyFromHost(outlier, len);
    // #else
    auto outlier_bin = io::ReadBinaryFile<uint8_t>(fi_outlier_as_cuspm, nnz_outlier * (sizeof(int) + sizeof(float)));
    auto outlier_csrColIndC = reinterpret_cast<int*>(outlier_bin);
    auto outlier_csrValC = reinterpret_cast<float*>(outlier_bin + nnz_outlier * sizeof(int));  // TODO template
    auto outlier     = new T[len]();
    for (auto i = 0; i < nnz_outlier; i++) outlier[outlier_csrColIndC[i]] = outlier_csrValC[i];
    auto d_outlier = mem::CreateDeviceSpaceAndMemcpyFromHost(outlier, len);
    // #endif
    */

    // clang-format off
    auto bytelen_csrRowPtrC  = sizeof(int)   * (padded_edge + 1);
    auto bytelen_csrColIndC  = sizeof(int)   *  nnz_outlier;
    auto bytelen_csrValC     = sizeof(float) *  nnz_outlier;
    auto bytelen_outlier_bin = bytelen_csrRowPtrC + bytelen_csrColIndC + bytelen_csrValC;
    auto outlier_bin         = io::ReadBinaryFile<uint8_t>(fi_outlier_as_cuspm, bytelen_outlier_bin);
    auto outlier_csrRowPtrC  = reinterpret_cast<int*  >(outlier_bin);
    auto outlier_csrColIndC  = reinterpret_cast<int*  >(outlier_bin + bytelen_csrRowPtrC);
    auto outlier_csrValC     = reinterpret_cast<float*>(outlier_bin + bytelen_csrRowPtrC + bytelen_csrColIndC);  // TODO template
    // clang-format on

    cout << log_dbg << "outlier_bin byte length:\t" << bytelen_outlier_bin << endl;
    cout << log_info << "Extracting outlier (from CSR format)..." << endl;

    for (auto i = 0; i < nnz_outlier + 1; i++) cout << outlier_csrRowPtrC[i] << endl;

    auto outlier = new T[padded_len]();
    auto gi      = 0;
    auto irow    = 0;
    auto lda     = padded_edge;
    while (irow < nnz_outlier) {
        cout << "irow\t" << irow << endl;
        while (gi < outlier_csrRowPtrC[irow + 1]) {
            outlier[lda * irow + outlier_csrColIndC[gi]] = outlier_csrValC[gi];
            gi++;
        }
        irow++;
    }

    cout << log_info << "Finished extracting outlier (from CSR format)..." << endl;

    // TODO merge d_outlier and d_data
    auto d_outlier = mem::CreateDeviceSpaceAndMemcpyFromHost(outlier, len);  // okay to use the original len
    auto d_xdata   = mem::CreateCUDASpace<T>(len);
    ReversedPdQ(d_xdata, d_bcode, d_outlier, dims_L16, ebs_L4[EBx2]);
    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, len);

    cout << log_info << "Decompression finished.\n\n";

    // TODO move CR out of VerifyData
    auto   odata        = io::ReadBinaryFile<T>(fi, len);
    size_t archive_size = 0;
    // TODO huffman chunking metadata
    if (not ap->skip_huffman)
        archive_size += total_uInt * sizeof(H)    // Huffman coded
                        + huffman_metadata_size;  // chunking metadata and reverse codebook
    else
        archive_size += len * sizeof(Q);
    archive_size += nnz_outlier * (sizeof(T) + sizeof(int));

    // TODO g++ and clang++ use mangled type_id name, add macro
    // https://stackoverflow.com/a/4541470/8740097
    auto demangle = [](const char* name) {
        int               status         = -4;
        char*             res            = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        const char* const demangled_name = (status == 0) ? res : name;
        string            ret_val(demangled_name);
        free(res);
        return ret_val;
    };

    if (ap->skip_huffman) {
        cout << log_info << "dtype is \""         //
             << demangle(typeid(T).name())        // demangle
             << "\", and quant. code type is \""  //
             << demangle(typeid(Q).name())        // demangle
             << "\"; a CR of no greater than "    //
             << (sizeof(T) / sizeof(Q)) << " is expected when Huffman codec is skipped." << endl;
    }

    if (ap->pre_binning) cout << log_info << "Because of binning (2x2), we have a 4x CR as the normal case." << endl;
    if (not ap->skip_huffman) {
        cout << log_info << "Huffman metadata of chunking and reverse codebook size (in bytes): " << huffman_metadata_size << endl;
        cout << log_info << "Huffman coded output size: " << total_uInt * sizeof(H) << endl;
    }

    analysis::VerifyData(
        xdata, odata, len,  //
        false,              //
        ebs_L4[EB],         //
        archive_size,
        ap->pre_binning ? 4 : 1);  // suppose binning is 2x2

    if (!ap->skip_writex) {
        if (!ap->alt_xout_name.empty()) cout << log_info << "Default decompressed data is renamed from " << string(fi + ".szx") << " to " << f_extract << endl;
        io::WriteBinaryFile(xdata, len, &f_extract);
    }

    // clean up
    delete[] odata;
    delete[] xdata;
    delete[] outlier;
    delete[] xbcode;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_bcode);
}

template void cuSZ::workflow::Compress<float, uint8_t, uint32_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint8_t, uint64_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint16_t, uint32_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint16_t, uint64_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint32_t, uint32_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint32_t, uint64_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);

template void cuSZ::workflow::Decompress<float, uint8_t, uint32_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint8_t, uint64_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint16_t, uint32_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint16_t, uint64_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint32_t, uint32_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint32_t, uint64_t>(string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);

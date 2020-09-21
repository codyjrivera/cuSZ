/**
 * @file par_huffman.cu
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Parallel Huffman Construction to generates canonical forward codebook.
 *        Based on [Ostadzadeh et al. 2007] (https://dblp.org/rec/conf/pdpta/OstadzadehEZMB07.bib)
 *        "A Two-phase Practical Parallel Algorithm for Construction of Huffman Codes".
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2020-05
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "cuda_error_handling.cuh"
#include "cuda_mem.cuh"
#include "dbg_gpu_printing.cuh"
#include "format.hh"
#include "par_huffman.cuh"
#include "par_merge.cuh"

// Helper implementations
template <typename T>
__global__ void GPU_FillArraySequence(T* array, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread < size) { array[thread] = thread; }
}

// Precondition -- Result is preset to be equal to size
template <typename T>
__global__ void GPU_GetFirstNonzeroIndex(T* array, unsigned int size, unsigned int* result)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (array[thread] != 0) { atomicMin(result, thread); }
}

__global__ void GPU_GetMaxCWLength(unsigned int* CL, unsigned int size, unsigned int* result)
{
    (void)size;
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread == 0) { *result = CL[0]; }
}

// Reorders given a set of indices. Programmer must ensure that all index[i]
// are unique or else race conditions may occur
template <typename T, typename Q>
__global__ void GPU_ReorderByIndex(T* array, Q* index, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    T            temp;
    Q            newIndex;
    if (thread < size) {
        temp            = array[thread];
        newIndex        = index[thread];
        array[newIndex] = temp;
    }
}

// Reverses a given array.
template <typename T>
__global__ void GPU_ReverseArray(T* array, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread < size / 2) {
        T temp                   = array[thread];
        array[thread]            = array[size - thread - 1];
        array[size - thread - 1] = temp;
    }
}

// Parallel codebook generation wrapper
template <typename Q, typename H>
void ParGetCodebook(int dict_size, unsigned int* _d_freq, H* _d_codebook, uint8_t* _d_decode_meta)
{
    // Metadata
    auto type_bw  = sizeof(H) * 8;
    auto _d_first = reinterpret_cast<H*>(_d_decode_meta);
    auto _d_entry = reinterpret_cast<H*>(_d_decode_meta + (sizeof(H) * type_bw));
    auto _d_qcode = reinterpret_cast<Q*>(_d_decode_meta + (sizeof(H) * 2 * type_bw));

    // Sort Qcodes by frequency
    int nblocks = (dict_size / 1024) + 1;
    GPU_FillArraySequence<Q><<<nblocks, 1024>>>(_d_qcode, (unsigned int)dict_size);
    cudaDeviceSynchronize();

    SortByFreq(_d_freq, _d_qcode, dict_size);
    cudaDeviceSynchronize();

    unsigned int* d_first_nonzero_index;
    unsigned int  first_nonzero_index = dict_size;
    cudaMalloc(&d_first_nonzero_index, sizeof(unsigned int));
    cudaMemcpy(d_first_nonzero_index, &first_nonzero_index, sizeof(unsigned int), cudaMemcpyHostToDevice);
    GPU_GetFirstNonzeroIndex<unsigned int><<<nblocks, 1024>>>(_d_freq, dict_size, d_first_nonzero_index);
    cudaDeviceSynchronize();
    cudaMemcpy(&first_nonzero_index, d_first_nonzero_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_first_nonzero_index);

    int           nz_dict_size   = dict_size - first_nonzero_index;
    unsigned int* _nz_d_freq     = _d_freq + first_nonzero_index;
    H*            _nz_d_codebook = _d_codebook + first_nonzero_index;
    int           nz_nblocks     = (nz_dict_size / 1024) + 1;

    // Memory Allocation -- Perhaps put in another wrapper
    // clang-format off
    unsigned int *CL         = nullptr;
    /*unsigned int* lNodesFreq*/         int *lNodesLeader = nullptr;
    unsigned int *iNodesFreq = nullptr;  int *iNodesLeader = nullptr;
    unsigned int *tempFreq   = nullptr;  int *tempIsLeaf   = nullptr;  int *tempIndex = nullptr;
    unsigned int *copyFreq   = nullptr;  int *copyIsLeaf   = nullptr;  int *copyIndex = nullptr;
    cudaMalloc(&CL,           nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&lNodesLeader, nz_dict_size * sizeof(int)          );
    cudaMalloc(&iNodesFreq,   nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&iNodesLeader, nz_dict_size * sizeof(int)          );
    cudaMalloc(&tempFreq,     nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&tempIsLeaf,   nz_dict_size * sizeof(int)          );
    cudaMalloc(&tempIndex,    nz_dict_size * sizeof(int)          );
    cudaMalloc(&copyFreq,     nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&copyIsLeaf,   nz_dict_size * sizeof(int)          );
    cudaMalloc(&copyIndex,    nz_dict_size * sizeof(int)          );
    cudaMemset(CL, 0,         nz_dict_size * sizeof(int)          );
    // clang-format on

    // Merge configuration -- Change for V100
    int ELTS_PER_SEQ_MERGE = 16;

    int       mblocks  = (nz_dict_size / ELTS_PER_SEQ_MERGE) + 1;
    int       mthreads = 32;
    uint32_t* diagonal_path_intersections;
    cudaMalloc(&diagonal_path_intersections, (2 * (mblocks + 1)) * sizeof(uint32_t));

    // Codebook already init'ed
    cudaDeviceSynchronize();

    // Call first kernel
    // Collect arguments
    void* CL_Args[] = {(void*)&_nz_d_freq,   (void*)&CL,
                       (void*)&nz_dict_size, (void*)&_nz_d_freq,
                       (void*)&lNodesLeader, (void*)&iNodesFreq,
                       (void*)&iNodesLeader, (void*)&tempFreq,
                       (void*)&tempIsLeaf,   (void*)&tempIndex,
                       (void*)&copyFreq,     (void*)&copyIsLeaf,
                       (void*)&copyIndex,    (void*)&diagonal_path_intersections,
                       (void*)&mblocks,      (void*)&mthreads};
    // Cooperative Launch
    cudaLaunchCooperativeKernel(
        (void*)parHuff::GPU_GenerateCL<unsigned int>, mblocks, mthreads, CL_Args,
        5 * sizeof(int32_t) + 32 * sizeof(int32_t));
    cudaDeviceSynchronize();

    // Exits if the highest codeword length is greater than what
    // the adaptive representation can handle
    // TODO do  proper cleanup

    unsigned int* d_max_CL;
    unsigned int  max_CL;
    cudaMalloc(&d_max_CL, sizeof(unsigned int));
    GPU_GetMaxCWLength<<<1, 1>>>(CL, nz_dict_size, d_max_CL);
    cudaDeviceSynchronize();
    cudaMemcpy(&max_CL, d_max_CL, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_first_nonzero_index);

    int max_CW_bits = (sizeof(H) * 8) - 8;
    if (max_CL > max_CW_bits) {
        cout << log_err << "Cannot store all Huffman codewords in " << max_CW_bits + 8 << "-bit representation" << endl;
        cout << log_err << "Huffman codeword representation requires at least " << max_CL + 8
             << " bits (longest codeword: " << max_CL << " bits)" << endl;
        cout << log_err << "(Consider running with -H 64)" << endl << endl;
        cout << log_err << "Exiting cuSZ ..." << endl;
        exit(1);
    }

    void* CW_Args[] = {
        (void*)&CL,              //
        (void*)&_nz_d_codebook,  //
        (void*)&_d_first,        //
        (void*)&_d_entry,        //
        (void*)&nz_dict_size};

    // Call second kernel
    cudaLaunchCooperativeKernel(
        (void*)parHuff::GPU_GenerateCW<unsigned int, H>,  //
        nz_nblocks,                                       //
        1024,                                             //
        CW_Args);
    cudaDeviceSynchronize();

#ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(_d_codebook, dict_size);  // PASS
    cudaDeviceSynchronize();
#endif

    // Reverse _d_qcode and _d_codebook
    GPU_ReverseArray<H><<<nblocks, 1024>>>(_d_codebook, (unsigned int)dict_size);
    GPU_ReverseArray<Q><<<nblocks, 1024>>>(_d_qcode, (unsigned int)dict_size);
    cudaDeviceSynchronize();

    GPU_ReorderByIndex<H, Q><<<nblocks, 1024>>>(_d_codebook, _d_qcode, (unsigned int)dict_size);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(CL);
    cudaFree(lNodesLeader);
    cudaFree(iNodesFreq);
    cudaFree(iNodesLeader);
    cudaFree(tempFreq);
    cudaFree(tempIsLeaf);
    cudaFree(tempIndex);
    cudaFree(copyFreq);
    cudaFree(copyIsLeaf);
    cudaFree(copyIndex);
    cudaFree(diagonal_path_intersections);
    cudaDeviceSynchronize();

#ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(_d_codebook, dict_size);  // PASS
    cudaDeviceSynchronize();
#endif
}

// Specialize wrapper
template void ParGetCodebook<uint8_t, uint32_t>(int dict_size, unsigned int* freq, uint32_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint8_t, uint64_t>(int dict_size, unsigned int* freq, uint64_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint16_t, uint32_t>(int dict_size, unsigned int* freq, uint32_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint16_t, uint64_t>(int dict_size, unsigned int* freq, uint64_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint32_t, uint32_t>(int dict_size, unsigned int* freq, uint32_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint32_t, uint64_t>(int dict_size, unsigned int* freq, uint64_t* codebook, uint8_t* meta);

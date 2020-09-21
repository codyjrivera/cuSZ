/**
 * @file par_huffman_gencw.cu
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Parallel Huffman Construction codeword generation
 *        Based on [Ostadzadeh et al. 2007] (https://dblp.org/rec/conf/pdpta/OstadzadehEZMB07.bib)
 *        "A Two-phase Practical Parallel Algorithm for Construction of Huffman Codes".
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2020-05
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National L                                      aboratory
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



// Parallel huffman global memory and kernels
namespace parHuff {
// GenerateCW Locals
    __device__ int CCL;
    __device__ int CDPI;
    __device__ int newCDPI;

}

template <typename F, typename H>
__global__ void parHuff::GPU_GenerateCW(F* CL, H* CW, H* first, H* entry, int size)
{
    unsigned int       thread       = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int i            = thread;  // Porting convenience
    auto               current_grid = this_grid();
    auto               type_bw      = sizeof(H) * 8;

    /* Reverse in place - Probably a more CUDA-appropriate way */
    if (thread < size / 2) {
        F temp           = CL[i];
        CL[i]            = CL[size - i - 1];
        CL[size - i - 1] = temp;
    }
    current_grid.sync();

    if (thread == 0) {
        CCL        = CL[0];
        CDPI       = 0;
        newCDPI    = size - 1;
        entry[CCL] = 0;

        // Edge case -- only one input symbol
        CW[CDPI]       = 0;
        first[CCL]     = CW[CDPI] ^ (((H)1 << (H)CL[CDPI]) - 1);
        entry[CCL + 1] = 1;
    }
    current_grid.sync();

    // Initialize first and entry arrays
    if (thread < CCL) {
        // Initialization of first to Max ensures that unused code
        // lengths are skipped over in decoding.
        first[i] = std::numeric_limits<H>::max();
        entry[i] = 0;
    }
    // Initialize first element of entry
    current_grid.sync();

    while (CDPI < size - 1) {
        // CDPI update
        if (i < size - 1 && CL[i + 1] > CCL) { atomicMin(&newCDPI, i); }
        current_grid.sync();

        // Last element to update
        const int updateEnd = (newCDPI >= size - 1) ? type_bw : CL[newCDPI + 1];
        // Fill base
        const int curEntryVal = entry[CCL];
        // Number of elements of length CCL
        const int numCCL = (newCDPI - CDPI + 1);

        // Get first codeword
        if (i == 0) {
            if (CDPI == 0) { CW[newCDPI] = 0; }
            else {
                CW[newCDPI] = CW[CDPI];  // Pre-stored
            }
        }
        current_grid.sync();

        if (i < size) {
            // Parallel canonical codeword generation
            if (i >= CDPI && i < newCDPI) { CW[i] = CW[newCDPI] + (newCDPI - i); }
        }

        // Update entry and first arrays in O(1) time
        if (thread > CCL && thread < updateEnd) { entry[i] = curEntryVal + numCCL; }
        // Add number of entries to next CCL
        if (thread == 0) {
            if (updateEnd < type_bw) { entry[updateEnd] = curEntryVal + numCCL; }
        }
        current_grid.sync();

        // Update first array in O(1) time
        if (thread == CCL) {
            // Flip least significant CL[CDPI] bits
            first[CCL] = CW[CDPI] ^ (((H)1 << (H)CL[CDPI]) - 1);
        }
        if (thread > CCL && thread < updateEnd) { first[i] = std::numeric_limits<H>::max(); }
        current_grid.sync();

        if (thread == 0) {
            if (newCDPI < size - 1) {
                int CLDiff = CL[newCDPI + 1] - CL[newCDPI];
                // Add and shift -- Next canonical code
                CW[newCDPI + 1] = ((CW[CDPI] + 1) << CLDiff);
                CCL             = CL[newCDPI + 1];

                ++newCDPI;
            }

            // Update CDPI to newCDPI after codeword length increase
            CDPI    = newCDPI;
            newCDPI = size - 1;
        }
        current_grid.sync();
    }

    if (thread < size) {
        /* Make encoded codeword compatible with CUSZ */
        CW[i] = (CW[i] | (((H)CL[i] & (H)0xffu) << ((sizeof(H) * 8) - 8))) ^ (((H)1 << (H)CL[i]) - 1);
    }
    current_grid.sync();

    /* Reverse partial codebook */
    if (thread < size / 2) {
        H temp           = CW[i];
        CW[i]            = CW[size - i - 1];
        CW[size - i - 1] = temp;
    }
}


template
__global__ void parHuff::GPU_GenerateCW<unsigned int, uint32_t>(unsigned int* CL, 
                                                                uint32_t* CW, uint32_t* first, uint32_t* entry, 
                                                                int size);
template
__global__ void parHuff::GPU_GenerateCW<unsigned int, uint64_t>(unsigned int* CL, 
                                                                uint64_t* CW, uint64_t* first, uint64_t* entry, 
                                                                int size);

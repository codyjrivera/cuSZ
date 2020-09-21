/**
 * @file par_huffman.cu
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Parallel Huffman Construction codeword length generation
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


// Mathematically correct mod
#define MOD(a, b) ((((a) % (b)) + (b)) % (b))

namespace parHuff {
    // GenerateCL Locals
    __device__ int iNodesFront = 0;
    __device__ int iNodesRear  = 0;
    __device__ int lNodesCur   = 0;

    __device__ int iNodesSize = 0;
    __device__ int curLeavesNum;

    __device__ int minFreq;

    __device__ int tempLength;

    __device__ int mergeFront;
    __device__ int mergeRear;

    __device__ int lNodesIndex;
}

// clang-format off
template <typename F>
__global__ void parHuff::GPU_GenerateCL(
    F*  histogram,  F* CL,  int size,
    /* Global Arrays */
    F* lNodesFreq,  int* lNodesLeader,
    F* iNodesFreq,  int* iNodesLeader,
    F* tempFreq,    int* tempIsLeaf,    int* tempIndex,
    F* copyFreq,    int* copyIsLeaf,    int* copyIndex,
    uint32_t* diagonal_path_intersections, int mblocks, int mthreads)
{
    // clang-format on

    extern __shared__ int32_t shmem[];
    // Shared variables
    int32_t& x_top     = shmem[0];
    int32_t& y_top     = shmem[1];
    int32_t& x_bottom  = shmem[2];
    int32_t& y_bottom  = shmem[3];
    int32_t& found     = shmem[4];
    int32_t* oneorzero = &shmem[5];

    unsigned int       thread       = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int i            = thread;  // Adaptation for easier porting
    auto               current_grid = this_grid();

    /* Initialization */
    if (thread < size) {
        lNodesLeader[i] = -1;
        CL[i]           = 0;
    }

    if (thread == 0) {
        iNodesFront = 0;
        iNodesRear  = 0;
        lNodesCur   = 0;

        iNodesSize = 0;
    }
    current_grid.sync();

    /* While there is not exactly one internal node */
    while (lNodesCur < size || iNodesSize > 1) {
        /* Combine two most frequent nodes on same level */
        if (thread == 0) {
            F   midFreq[4];
            int midIsLeaf[4];
            for (int i = 0; i < 4; ++i) midFreq[i] = UINT_MAX;

            if (lNodesCur < size) {
                midFreq[0]   = lNodesFreq[lNodesCur];
                midIsLeaf[0] = 1;
            }
            if (lNodesCur < size - 1) {
                midFreq[1]   = lNodesFreq[lNodesCur + 1];
                midIsLeaf[1] = 1;
            }
            if (iNodesSize >= 1) {
                midFreq[2]   = iNodesFreq[iNodesFront];
                midIsLeaf[2] = 0;
            }
            if (iNodesSize >= 2) {
                midFreq[3]   = iNodesFreq[MOD(iNodesFront + 1, size)];
                midIsLeaf[3] = 0;
            }

            /* Select the minimum of minimums - 4elt sorting network */
            /* TODO There's likely a good 1-warp faster way to do this */
            {
                F   tempFreq;
                int tempIsLeaf;
                if (midFreq[1] > midFreq[3]) {
                    tempFreq     = midFreq[1];
                    midFreq[1]   = midFreq[3];
                    midFreq[3]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[1];
                    midIsLeaf[1] = midIsLeaf[3];
                    midIsLeaf[3] = tempIsLeaf;
                }
                if (midFreq[0] > midFreq[2]) {
                    tempFreq     = midFreq[0];
                    midFreq[0]   = midFreq[2];
                    midFreq[2]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[0];
                    midIsLeaf[0] = midIsLeaf[2];
                    midIsLeaf[2] = tempIsLeaf;
                }
                if (midFreq[0] > midFreq[1]) {
                    tempFreq     = midFreq[0];
                    midFreq[0]   = midFreq[1];
                    midFreq[1]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[0];
                    midIsLeaf[0] = midIsLeaf[1];
                    midIsLeaf[1] = tempIsLeaf;
                }
                if (midFreq[2] > midFreq[3]) {
                    tempFreq     = midFreq[2];
                    midFreq[2]   = midFreq[3];
                    midFreq[3]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[2];
                    midIsLeaf[2] = midIsLeaf[3];
                    midIsLeaf[3] = tempIsLeaf;
                }
                if (midFreq[1] > midFreq[2]) {
                    tempFreq     = midFreq[1];
                    midFreq[1]   = midFreq[2];
                    midFreq[2]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[1];
                    midIsLeaf[1] = midIsLeaf[2];
                    midIsLeaf[2] = tempIsLeaf;
                }
            }

            minFreq = midFreq[0];
            if (midFreq[1] < UINT_MAX) { minFreq += midFreq[1]; }
            iNodesFreq[iNodesRear]   = minFreq;
            iNodesLeader[iNodesRear] = -1;

            /* If is leaf */
            if (midIsLeaf[0]) {
                lNodesLeader[lNodesCur] = iNodesRear;
                ++CL[lNodesCur], ++lNodesCur;
            }
            else {
                iNodesLeader[iNodesFront] = iNodesRear;
                iNodesFront               = MOD(iNodesFront + 1, size);
            }
            if (midIsLeaf[1]) {
                lNodesLeader[lNodesCur] = iNodesRear;
                ++CL[lNodesCur], ++lNodesCur;
            }
            else {
                iNodesLeader[iNodesFront] = iNodesRear;
                iNodesFront               = MOD(iNodesFront + 1, size); /* ? */
            }

            // iNodesRear = MOD(iNodesRear + 1, size);

            iNodesSize = MOD(iNodesRear - iNodesFront, size);
        }

        // int curLeavesNum;
        /* Select elements to copy -- parallelized */
        curLeavesNum = 0;
        current_grid.sync();
        if (i >= lNodesCur && i < size) {
            // Parallel component
            int threadCurLeavesNum;
            if (lNodesFreq[i] <= minFreq) {
                threadCurLeavesNum = i - lNodesCur + 1;
                // Atomic max -- Largest valid index
                atomicMax(&curLeavesNum, threadCurLeavesNum);
            }

            if (i - lNodesCur < curLeavesNum) {
                copyFreq[i - lNodesCur]   = lNodesFreq[i];
                copyIndex[i - lNodesCur]  = i;
                copyIsLeaf[i - lNodesCur] = 1;
            }
        }

        current_grid.sync();

        /* Updates Iterators */
        if (thread == 0) {
            mergeRear  = iNodesRear;
            mergeFront = iNodesFront;

            if ((curLeavesNum + iNodesSize) % 2 == 0) { iNodesFront = iNodesRear; }
            /* Odd number of nodes to merge - leave out one*/
            else if (
                (iNodesSize != 0)                                                                        //
                and (curLeavesNum == 0                                                                   //
                     or (histogram[lNodesCur + curLeavesNum] <= iNodesFreq[MOD(iNodesRear - 1, size)]))  //
            ) {
                mergeRear   = MOD(mergeRear - 1, size);
                iNodesFront = MOD(iNodesRear - 1, size);
            }
            else {
                iNodesFront = iNodesRear;
                --curLeavesNum;
            }

            lNodesCur  = lNodesCur + curLeavesNum;
            iNodesRear = MOD(iNodesRear + 1, size);
        }
        current_grid.sync();

        /* Parallelized Merging Phase */

        /*if (thread == 0) {
            merge(copyFreq, copyIndex, copyIsLeaf, 0, curLeavesNum,
                    iNodesFreq, mergeFront, mergeRear, size,
                    tempFreq, tempIndex, tempIsLeaf, tempLength);
                    }*/

        parMerge(
            copyFreq, copyIndex, copyIsLeaf, 0, curLeavesNum,  //
            iNodesFreq, mergeFront, mergeRear, size,           //
            tempFreq, tempIndex, tempIsLeaf, tempLength,       //
            diagonal_path_intersections, mblocks, mthreads,    //
            x_top, y_top, x_bottom, y_bottom, found, oneorzero);
        current_grid.sync();

        /* Melding phase -- New */
        if (thread < tempLength / 2) {
            int ind           = MOD(iNodesRear + i, size);
            iNodesFreq[ind]   = tempFreq[(2 * i)] + tempFreq[(2 * i) + 1];
            iNodesLeader[ind] = -1;

            if (tempIsLeaf[(2 * i)]) {
                lNodesLeader[tempIndex[(2 * i)]] = ind;
                ++CL[tempIndex[(2 * i)]];
            }
            else {
                iNodesLeader[tempIndex[(2 * i)]] = ind;
            }
            if (tempIsLeaf[(2 * i) + 1]) {
                lNodesLeader[tempIndex[(2 * i) + 1]] = ind;
                ++CL[tempIndex[(2 * i) + 1]];
            }
            else {
                iNodesLeader[tempIndex[(2 * i) + 1]] = ind;
            }
        }
        current_grid.sync();

        if (thread == 0) { iNodesRear = MOD(iNodesRear + (tempLength / 2), size); }
        current_grid.sync();

        /* Update leaders */
        if (thread < size) {
            if (lNodesLeader[i] != -1) {
                if (iNodesLeader[lNodesLeader[i]] != -1) {
                    lNodesLeader[i] = iNodesLeader[lNodesLeader[i]];
                    ++CL[i];
                }
            }
        }
        current_grid.sync();

        if (thread == 0) { iNodesSize = MOD(iNodesRear - iNodesFront, size); }
        current_grid.sync();
    }
}

template
__global__ void parHuff::GPU_GenerateCL<unsigned int>(
    unsigned int*  histogram,  unsigned int* CL,  int size,
    unsigned int* lNodesFreq,  int* lNodesLeader,
    unsigned int* iNodesFreq,  int* iNodesLeader,
    unsigned int* tempFreq,    int* tempIsLeaf,    int* tempIndex,
    unsigned int* copyFreq,    int* copyIsLeaf,    int* copyIndex,
    uint32_t* diagonal_path_intersections, int mblocks, int mthreads);

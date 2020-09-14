#ifndef CUSZ_WORKFLOW_CUH
#define CUSZ_WORKFLOW_CUH

#include "argparse.hh"

namespace cuSZ {
namespace workflow {

template <typename T, typename Q, typename H>
void Compress(
    std::string& fi,  //
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      n_bits,
    size_t&      n_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap);

template <typename T, typename Q, typename H>
void Decompress(
    std::string& fi,
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      total_bits,
    size_t&      total_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap);

}  // namespace workflow

namespace impl {

inline size_t GetEdgeOfReinterpretedSquare(size_t l)
{
    auto sl = static_cast<size_t>(sqrt(l));
    return ((sl - 1) / 2 + 1) * 2;
};

template <typename T, typename Q>
void PdQ(T*, Q*, size_t*, double*);

template <typename T, typename Q>
void ReversedPdQ(T*, Q*, T*, size_t*, double);

template <typename T>
void ArchiveAsSpM(T* d_data, size_t padded_len, size_t padded_edge, int* nnz_outlier, std::string*);

template <typename T>
T* ExtractFromSpM(size_t padded_len, size_t padded_edge, int* nnz_outlier, std::string*);

template <typename T, typename Q>
void VerifyHuffman(string const&, size_t, Q*, int, size_t*, double*);

}  // namespace impl

}  // namespace cuSZ

#endif

/**
 * @file test_huffbook.cu
 * @author Jiannan Tian, Cody Rivera
 * @brief
 * @version 0.3
 * @date 2021-04-29
 * (repurposed) 2022-02-09
 * 
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include "common.hh"
#include "utils.hh"
#include "wrapper/huffman_parbook.cuh"

//using cusz::FREQ = unsigned int;
#include "kernel/hist.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using Freq = cusz::FREQ;
using Codeword = uint32_t;
 
__global__ void dummy() { float data = threadIdx.x; }

template <typename E>
static uint32_t get_revbook_nbyte(int dict_size) {
    using BOOK = Codeword;
    using SYM  = E;
    
    constexpr auto TYPE_BITCOUNT = sizeof(BOOK) * 8;
    return sizeof(BOOK) * (2 * TYPE_BITCOUNT) + sizeof(SYM) * dict_size;
}

template <typename E>
void print_codebook(string filename, size_t len, int dict_size) {
    Capsule<E> quant_codes(len, "quantization codes");
    Capsule<Freq> hist(dict_size, "histogram");
    
    quant_codes.template alloc<cusz::LOC::HOST_DEVICE>()
        .template from_fs_to<cusz::LOC::HOST>(filename)
        .host2device();

    hist.template alloc<cusz::LOC::HOST_DEVICE>();

    float x;
    wrapper::get_frequency(
        quant_codes.template get<cusz::LOC::DEVICE>(),
        len,
        hist.template get<cusz::LOC::DEVICE>(),
        dict_size,
        x);

    Capsule<Codeword> book(dict_size, "forward codebook");
    book.template alloc<cusz::LOC::HOST_DEVICE>();
    
    Capsule<uint8_t> revbook(get_revbook_nbyte<E>(dict_size), "reverse codebook");
    revbook.template alloc<cusz::LOC::HOST_DEVICE>();
    
    lossless::par_get_codebook<E, Codeword>(
        dict_size,
        hist.template get<cusz::LOC::DEVICE>(),
        book.template get<cusz::LOC::DEVICE>(),
        revbook.template get<cusz::LOC::DEVICE>());

    book.device2host();

    Codeword* host_book = book.template get<cusz::LOC::HOST>();

    for (int i = 0; i < dict_size; ++i) {
        cout << i << " : " << host_book[i] << endl;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " qcodes len dict_size (u8|u16)" << endl;
        return 1;
    }

    size_t len = (size_t) atol(argv[2]);
    int dict_size = atoi(argv[3]);
    string type = string(argv[4]);
    
    if (type == string("u8")) {
        print_codebook<uint8_t>(string(argv[1]), len, dict_size);
    } else if (type == string("u16")) {
        print_codebook<uint16_t>(string(argv[1]), len, dict_size);
    } else {
        cerr << "Type must be u8 or u16" << endl;
        return 1;    
    }
    
    return 0;
}

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

//using cusz::FRErrorQ = unsigned int;
#include "kernel/hist.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using Freq = cusz::FREQ;
 
__global__ void dummy() { float data = threadIdx.x; }

template <typename T>
void print_by_type(T num, char sep = '_', char ending = '\n')
{
    for (size_t j = 0; j < sizeof(T) * CHAR_BIT; j++) {
        printf("%u", (num >> ((sizeof(T) * CHAR_BIT - 1) - j)) & 0x01u);
        if (j != 0 and j != sizeof(T) * CHAR_BIT - 1 and j % 8 == 7) printf("%c", sep);
    }
    printf("%c", ending);
}

template <typename T>
void print_code_only(T num, size_t bitwidth, char sep = '_', char ending = '\n')
{
    for (size_t j = 0; j < bitwidth; j++) {
        printf("%u", (num >> ((bitwidth - 1) - j)) & 0x01u);
        if (j != 0 and j != bitwidth - 1 and j % 8 == 7) printf("%c", sep);
    }
    printf("%c", ending);
}

template <typename T>
void snippet_print_bitset_full(T num) {
    print_by_type(num, '_', '\t');
    size_t bitwidth = *((uint8_t*)&num + sizeof(T) - 1);
    //    size_t code_bitwidth = ((static_cast<T>(0xffu) << (sizeof(T) * 8 - 8)) & num) >> (sizeof(T) * 8 - 8);
    printf("len: %3lu\tcode: ", bitwidth);
    print_code_only<T>(num, bitwidth, '\0', '\n');
}

template <typename T>
void print_codebook(T* codebook, size_t len) {
    printf("--------------------------------------------------------------------------------\n");
    printf("printing codebook\n");
    printf("--------------------------------------------------------------------------------\n");
    T buffer;
    for (size_t i = 0; i < len; i++) {
        buffer = codebook[i];
        if (buffer == ((T)0x0)) continue;
        printf("%5lu\t", i);
        snippet_print_bitset_full(buffer);
    }
    printf("--------------------------------------------------------------------------------\n");
    printf("done printing codebook\n");
    printf("--------------------------------------------------------------------------------\n");
}


template <typename Error, typename Codeword>
static uint32_t get_revbook_nbyte(int dict_size) {
    using BOOK = Codeword;
    using SYM  = Error;
    
    constexpr auto TYPError_BITCOUNT = sizeof(BOOK) * 8;
    return sizeof(BOOK) * (2 * TYPError_BITCOUNT) + sizeof(SYM) * dict_size;
}

template <typename Error, typename Codeword>
void test_codebook(string filename, size_t len, int dict_size) {
    Capsule<Error> quant_codes(len, "quantization codes");
    Capsule<Freq> hist(dict_size, "histogram");
    
    quant_codes.template alloc<cusz::LOC::HOST_DEVICE>()
        .template from_file<cusz::LOC::HOST>(filename)
        .host2device();

    hist.template alloc<cusz::LOC::HOST_DEVICE>();

    float x;
    kernel_wrapper::get_frequency(
        quant_codes.template get<cusz::LOC::DEVICE>(),
        len,
        hist.template get<cusz::LOC::DEVICE>(),
        dict_size,
        x,
        0);

    Capsule<Codeword> book(dict_size, "forward codebook");
    book.template alloc<cusz::LOC::HOST_DEVICE>();
    
    Capsule<uint8_t> revbook(get_revbook_nbyte<Error, Codeword>(dict_size), "reverse codebook");
    revbook.template alloc<cusz::LOC::HOST_DEVICE>();
    
    kernel_wrapper::par_get_codebook<Error, Codeword>(
        hist.template get<cusz::LOC::DEVICE>(),
        dict_size,
        book.template get<cusz::LOC::DEVICE>(),
        revbook.template get<cusz::LOC::DEVICE>(),
        0);

    book.device2host();

    Codeword* host_book = book.template get<cusz::LOC::HOST>();

    print_codebook(host_book, dict_size);
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        cerr << "Usage: " << argv[0] << " qcodes len dict_size [(u8|u16) [u32|u64]]" << endl;
        return 1;
    }

    string file_name = string(argv[1]);
    size_t len = (size_t) atol(argv[2]);
    int dict_size = atoi(argv[3]);

    string error_type = string("u16");
    if (argc >= 5) {
        error_type = string(argv[4]);
    }

    string codeword_type = string("u32");
    if (argc >= 6) {
        codeword_type = string(argv[5]);
    }

    if (error_type == "u8") {
        if (codeword_type == "u32") {
            test_codebook<uint8_t, uint32_t>(file_name, len, dict_size);
        } else if (codeword_type == "u64") {
            test_codebook<uint8_t, unsigned long long>(file_name, len, dict_size);
        } else {
            cerr << "CodebookType must be u32 or u64" << endl;
            return 1;
        }
    } else if (error_type == "u16") {
        if (codeword_type == "u32") {
            test_codebook<uint16_t, uint32_t>(file_name, len, dict_size);
        } else if (codeword_type == "u64") {
            test_codebook<uint16_t, unsigned long long>(file_name, len, dict_size);
        } else {
            cerr << "CodebookType must be u32 or u64" << endl;
            return 1;
        }
    } else {
        cerr << "ErrorType must be u8 or u16" << endl;
        return 1;    
    }
    
    return 0;
}

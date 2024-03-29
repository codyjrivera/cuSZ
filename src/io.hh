#ifndef IO_HH
#define IO_HH

/**
 * @file io.hh
 * @author Jiannan Tian
 * @brief Read and write binary.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2019-08-27
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <fstream>
#include <iostream>

namespace io {

template <typename T>
T* ReadBinaryFile(const std::string& __name, size_t __len)
{
    std::ifstream ifs(__name.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << __name << std::endl;
        exit(1);
        // return;
    }
    auto __a = new T[__len]();
    ifs.read(reinterpret_cast<char*>(__a), std::streamsize(__len * sizeof(T)));
    ifs.close();
    return __a;
}

template <typename T>
T* ReadBinaryFile(const std::string& __name, T* __a, size_t __len)
{
    std::ifstream ifs(__name.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << __name << std::endl;
        exit(1);
        // return;
    }
    // auto __a = new T[__len]();
    ifs.read(reinterpret_cast<char*>(__a), std::streamsize(__len * sizeof(T)));
    ifs.close();
    return __a;
}

template <typename T>
void WriteBinaryFile(T* const __a, size_t const __len, std::string const* const __name)
{
    std::ofstream ofs(__name->c_str(), std::ios::binary | std::ios::out);
    if (not ofs.is_open()) return;
    ofs.write(reinterpret_cast<const char*>(__a), std::streamsize(__len * sizeof(T)));
    ofs.close();
}

}  // namespace io

#endif  // IO_HH

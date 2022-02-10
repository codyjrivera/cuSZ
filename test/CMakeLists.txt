cmake_minimum_required(VERSION 3.18)

project(cusz_asap CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

## uncomment for pre-Turing
# set(CMAKE_CUDA_ARCHITECTURES 60 62 70 72)
## uncomment for V100
set(CMAKE_CUDA_ARCHITECTURES 70)
## uncomment for Turing
# set(CMAKE_CUDA_ARCHITECTURES 75)
## uncmoment for Ampere
# set(CMAKE_CUDA_ARCHITECTURES 80 86)
## uncmoment for A100
# set(CMAKE_CUDA_ARCHITECTURES 80)

enable_language(CUDA)

include_directories(../src)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

set(LIB_TYPE SHARED)

add_executable(huffbook src/test_huffbook.cu)
target_link_libraries(huffbook 
	${CMAKE_SOURCE_DIR}/../Release/libnvgpusz.a 
	${CMAKE_SOURCE_DIR}/../Release/libsp.a 
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
	${CMAKE_SOURCE_DIR}/../Release/libhuff.a 
	-lcusparse)

set_target_properties(huffbook PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

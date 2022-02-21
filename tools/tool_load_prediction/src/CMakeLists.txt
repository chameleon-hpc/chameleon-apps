# require a minimum version of cmake
cmake_minimum_required(VERSION 3.5)

# declare the project name
project(pred_model_tool)

# options for compiling with ITAC tracer
option(COMP_TRACE_ITAC "Enable tracing with Intel Trace Analyzer " off)
option(ENABLE_PYTORCH_CXX "Enable Pytorch-CXX to define ML/DL models" off)

# require Chameleon-lib, already loaded by module
# so, don't need to indicate it here

# require MPI_lib
# used mpiicc/mpiicpc to compile, so cmake could find
find_package(MPI REQUIRED)
set(MPI_CXX_LINK_FLAGS "-std=c++11 -fopenmp -lm -lchameleon -Wall -Wextra -O3 -DNDEBUG")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(DEFAULT_OPENMP_FLAG OpenMP::OpenMP_CXX)
endif()

# link mlpack, armadillo, boost_serialization
# find_package(MLPACK REQUIRED)
set(MLPACK_FLAGS "-larmadillo -lboost_serialization -lmlpack")

# create the library
add_library(tool SHARED tool.cpp)

# include VTune if the flag is on
if (COMP_TRACE_ITAC)
    include_directories($ENV{VT_ROOT}/include)
    link_directories($ENV{VT_ROOT}/lib)
    # set(ADDITIONAL_LIBS
    #     ${ADDITIONAL_LIBS}
    #     VT
    #     vtunwind
    #     dwarf
    #     elf
    # )
    # set_property(TARGET tool APPEND_STRING PROPERTY LINK_FLAGS " -DTRACE -trace ")
endif()

# inclde Pytorch CXX if the option is on
if (ENABLE_PYTORCH_CXX)
    find_package(Torch REQUIRED)
    set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# build the tool
target_link_libraries(tool ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${MLPACK_FLAGS} ${CMAKE_CXX_FLAGS} ${DEFAULT_OPENMP_FLAG} ${ADDITIONAL_LIBS})
if (ENABLE_PYTORCH_CXX)
    target_link_libraries(tool "${TORCH_LIBRARIES}")
endif()
set_property(TARGET tool PROPERTY CXX_STANDARD 17)
target_include_directories(tool PUBLIC ./)

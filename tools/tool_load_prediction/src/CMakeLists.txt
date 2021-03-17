# require a minimum version of cmake
cmake_minimum_required(VERSION 3.5)

# declare the project name
project(pred_model_tool)

# include Chameleon
set(CHAMELEON_LIB, "/dss/dsshome1/0A/di49mew/chameleon_tool_dev/install/chamtool_commthread/lib")
set(CHAMELEON_INC, "/dss/dsshome1/0A/di49mew/chameleon_tool_dev/install/chamtool_commthread/include")
include_directories(${CHAMELEON_INC})

# include MPI lib
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(MPI_CXX_LINK_FLAGS "-std=c++17 -lstdc++ -fopenmp -lm -lchameleon -L${CHAMELEON_LIB}")

# include VTune
# find_package(VTune REQUIRED)
# include_directories(${VT_INC})
# set(VT_FLAGS "-trace")

# include Torch lib
find_package(Torch REQUIRED)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_library(tool SHARED tool.cpp)
target_link_libraries(tool "${TORCH_LIBRARIES}")
target_link_libraries(tool ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${VT_FLAGS})  # remove BCL flags if we do not use
set_property(TARGET tool PROPERTY CXX_STANDARD 17)
target_include_directories(tool PUBLIC ./)

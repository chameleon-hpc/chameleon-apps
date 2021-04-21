# load new gcc version
module load gcc/9

# remove old compiled fiels
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake libtool.so  Makefile

# export chameleon lib and libffi
module use ~/.modules
module load chamtool_commthread
module load libffi-3.3
module load mlpack-3.4.2

# export itac
# export INCLUDE=/lrz/sys/intel/studio2019_u5/itac/2019.5.041/include:$INCLUDE
# export CPATH=/lrz/sys/intel/studio2019_u5/itac/2019.5.041/include:$CPATH

# choose the tool for samoa-chameleon
export SAMOA_EXAMPLE=1
export C_COMPILER=mpiicc
export CXX_COMPILER=mpiicpc

# run cmake
cmake -DCMAKE_PREFIX_PATH=/dss/dsshome1/0A/di49mew/loc-libs/libtorch \
        -DCMAKE_C_COMPILER=${C_COMPILER} \
        -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
        ../src/

# run build
cmake --build . --config Releas
# load new gcc version
module load gcc/9

# remove old compiled fiels
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake libtool.so  Makefile

# export chameleon lib and libffi
module use ~/.modules
module load chamtool_commthread
module load libffi-3.3

# export itac
# export INCLUDE=/lrz/sys/intel/studio2019_u5/itac/2019.5.041/include:$INCLUDE
# export CPATH=/lrz/sys/intel/studio2019_u5/itac/2019.5.041/include:$CPATH

# choose the tool for samoa-chameleon
export SAMOA_EXAMPLE=1

# run cmake
cmake -DCMAKE_PREFIX_PATH=/dss/dsshome1/0A/di49mew/loc-libs/libtorch -DCMAKE_C_COMPILER=/lrz/sys/compilers/gcc/9.2/bin/gcc -DCMAKE_CXX_COMPILER=/lrz/sys/compilers/gcc/9.2/bin/g++ -DMPI_CXX_COMPILER=/lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/lrzbin/mpicxx -DMPIEXEC_EXECUTABLE=/lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/bin/mpirun ../src/

# run build
cmake --build . --config Releas
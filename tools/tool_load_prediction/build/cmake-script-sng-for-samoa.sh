# remove old compiled fiels
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake libtool.so  Makefile

# export chameleon lib and libffi
module use ~/.modules
module load cmake/3.16.5
module load boost/1.73.0-gcc8
module load hwloc/2.2.0-gcc8
module load libffi-3.3-lrz-spack
module load cereal-1.3.0-gcc-8.4.0-iu6fggf-lrz-spack
module load hdf5/1.10.7-intel19-impi
module load armadillo-10.6.2 
module load ensmallen-2.17.0
module load mlpack-3.4.2        # built with oneapi/2021

# TODO: change mode depending on the experiments
module load chamtool_pred2_mig2_off1
# module load chamtool_pred2_mig2_off2

# choose the tool for samoa-chameleon
export SAMOA_EXAMPLE=1
export C_COMPILER=mpiicc
export CXX_COMPILER=mpiicpc

# export itac if we need
# but, maybe it's already loaded by intel oneapi/2021

# choose the tool for samoa-chameleon
echo "SAMOA_EXAMPLE=1"

# run cmake
# cmake -DCMAKE_PREFIX_PATH=/dss/dsshome1/lxc0D/ra56kop/local_libs/libtorch \
cmake -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        -DCMAKE_CXX_FLAGS="-DSAMOA_EXAMPLE=1" \
        ../src/

# run build
cmake --build . --config Releas
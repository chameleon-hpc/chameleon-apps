# load new gcc version
# TODO: if we don't use pytorch, icc/icpc should be fine
# on CoolMUC2, gcc version is 7.5.0, don't need another newer version here

# remove old compiled fiels
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake libtool.so  Makefile

# export chameleon lib and libffi
# intel-oneapi/2021 is already loaded on CoolMUC2
module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load boost-1.69.0-gcc-7.5.0-qvtxnfk # built with local-spack
module load chamtool_pred_mig
module load hwloc/2.0
module load set-hwloc-inc
module load libffi-3.3
module load hdf5-1.10.4         # built with oneapi/2021
module load armadillo-10.4.0    # built with oneapi/2021
module load ensmallen-2.16.2    # built with oneapi/2021
module load cereal
module load mlpack-3.4.2        # built with oneapi/2021

# export itac if we need
# but, maybe it's already loaded by intel oneapi/2021

# choose the tool for samoa-chameleon
export SAMOA_EXAMPLE=1

# run cmake
cmake -DCMAKE_PREFIX_PATH=/dss/dsshome1/lxc0D/ra56kop/local_libs/libtorch \
        -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        ../src/

# run build
cmake --build . --config Releas
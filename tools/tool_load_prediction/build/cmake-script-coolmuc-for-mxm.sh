# load new gcc version
# TODO: if we don't use pytorch, icc/icpc should be fine
# on CoolMUC2, gcc version is 7.5.0, don't need another newer version here

# remove old compiled fiels
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake libtool.so  Makefile

# export chameleon lib and libffi
# intel-oneapi/2021 is already loaded on CoolMUC2
module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load cmake-3.20.1-oneapi-2021.1-enbi76g
module load boost-1.76.0-gcc-7.5.0-fromrfo # built with local-spack
module load hwloc-2.4.1-oneapi-2021.1-rrkvihp
module load libffi-3.3-oneapi-2021.1-3vomiwb
module load cereal-1.3.0-gcc-7.5.0-jwb3bux # built by spack
module load hdf5-1.10.4         # built with oneapi/2021
module load armadillo-10.4.0    # built with oneapi/2021
module load ensmallen-2.16.2    # built with oneapi/2021
module load mlpack-3.4.2        # built with oneapi/2021
module load chamtool_pred3_mig1 # TODO: change mode depending on the experiments

# export itac if we need
# but, maybe it's already loaded by intel oneapi/2021

# choose the tool for samoa-chameleon
echo "MXM_EXAMPLE=1"

# run cmake
# cmake -DCMAKE_PREFIX_PATH=/dss/dsshome1/lxc0D/ra56kop/local_libs/libtorch \
cmake   -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        -DCMAKE_CXX_FLAGS="-DMXM_EXAMPLE=1" \
        ../src/

# run build
cmake --build . --config Releas
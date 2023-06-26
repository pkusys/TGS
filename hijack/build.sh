#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function build() {
    ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
    rm -rf ${ROOT}/build
    mkdir ${ROOT}/build
    cd ${ROOT}/build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

    rm -rf ${ROOT}/high-priority-lib 
    mkdir ${ROOT}/high-priority-lib
    cd ${ROOT}/high-priority-lib
    cp ${ROOT}/build/libcuda-control-high-priority.so ./libcuda-control.so

    touch ./ld.so.preload
    echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1\n/libnvidia-ml.so\n/libnvidia-ml.so.1" > ./ld.so.preload
    cp libcuda-control.so ./libnvidia-ml.so.1
    patchelf --set-soname libnvidia-ml.so.1 ./libnvidia-ml.so.1
    cp libcuda-control.so ./libnvidia-ml.so
    patchelf --set-soname libnvidia-ml.so ./libnvidia-ml.so
    cp libcuda-control.so ./libcuda.so.1
    patchelf --set-soname libcuda.so.1 ./libcuda.so.1
    cp libcuda-control.so ./libcuda.so
    patchelf --set-soname libcuda.so ./libcuda.so
    cp libcuda-control.so ./libcontroller.so
    patchelf --set-soname libcontroller.so ./libcontroller.so

    rm -rf ${ROOT}/low-priority-lib 
    mkdir ${ROOT}/low-priority-lib
    cd ${ROOT}/low-priority-lib
    cp ${ROOT}/build/libcuda-control-low-priority.so ./libcuda-control.so

    touch ./ld.so.preload
    echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1\n/libnvidia-ml.so\n/libnvidia-ml.so.1" > ./ld.so.preload
    cp libcuda-control.so ./libnvidia-ml.so.1
    patchelf --set-soname libnvidia-ml.so.1 ./libnvidia-ml.so.1
    cp libcuda-control.so ./libnvidia-ml.so
    patchelf --set-soname libnvidia-ml.so ./libnvidia-ml.so
    cp libcuda-control.so ./libcuda.so.1
    patchelf --set-soname libcuda.so.1 ./libcuda.so.1
    cp libcuda-control.so ./libcuda.so
    patchelf --set-soname libcuda.so ./libcuda.so
    cp libcuda-control.so ./libcontroller.so
    patchelf --set-soname libcontroller.so ./libcontroller.so

    cd ..
    # BASE_IMAGE=${BASE_IMAGE:-"tf_torch"}
    # docker build -t ${BASE_IMAGE} --network=host -f ./Dockerfile .
}

build
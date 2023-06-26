#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function run() {
    ./build.sh

    BASE_IMAGE=${BASE_IMAGE:-"tf_torch"}
    docker run --rm -it --gpus all --network=host --ipc=host \
        -v "$(pwd)/high-priority-lib/libcontroller.so:/libcontroller.so:ro" \
        -v "$(pwd)/high-priority-lib/libcuda.so:/libcuda.so:ro" \
        -v "$(pwd)/high-priority-lib/libcuda.so.1:/libcuda.so.1:ro" \
        -v "$(pwd)/high-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro" \
        -v "$(pwd)/high-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro" \
        -v "$(pwd)/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro" \
        -v "$(pwd)/gsharing:/etc/gsharing" \
        ${BASE_IMAGE} bash
}

run
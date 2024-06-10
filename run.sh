# LD_PRELOAD=./hijack/high-priority-lib/libcuda.so.1:./hijack/high-priority-lib/libcuda.so python3 ./workloads/matmul.py
cd hijack
./build.sh
cd -
./scripts/test_tgs.sh


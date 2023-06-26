set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null

trap "echo quit | sudo nvidia-cuda-mps-control" EXIT SIGTERM SIGINT

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
sudo nvidia-cuda-mps-control -d
python3 worker.py --trace config/test_mps.csv --log_path results/test_mps_results.csv
echo quit | sudo nvidia-cuda-mps-control

popd >/dev/null
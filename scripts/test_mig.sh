set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null

sudo nvidia-smi -i 1 -mig 1
sudo nvidia-smi mig -i 1 -cgi 5 -C
sudo nvidia-smi mig -i 1 -cgi 9 -C
python3 worker.py --trace config/test_mig.csv --log_path results/test_mig_results.csv
sudo nvidia-smi -mig 0

popd >/dev/null
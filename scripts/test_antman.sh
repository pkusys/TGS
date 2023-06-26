set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null

python3 worker.py --trace config/test_antman.csv --log_path results/test_antman_results.csv 2>&1 | tee backup_logs/test_antman.log

popd >/dev/null
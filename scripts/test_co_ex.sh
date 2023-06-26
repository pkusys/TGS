set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null

python3 worker.py --trace config/test_co_ex.csv --log_path results/test_co_ex_results.csv 2>&1 | tee backup_logs/test_co_ex.log

popd >/dev/null
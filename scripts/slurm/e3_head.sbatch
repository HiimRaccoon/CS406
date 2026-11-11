#!/usr/bin/env bash
#SBATCH --job-name=e3-head
#SBATCH --partition=normal
#SBATCH --account=uit
#SBATCH --qos=mig2-35
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384M
#SBATCH --time=09:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${CS406_REPO:?Set the accepted checkout}"
: "${CS406_PYTHON:?Set the reviewed Python executable}"
: "${CS406_E3_REQUEST:?Set one execution wrapper path}"
: "${CS406_E3_INVENTORY:?Set its byte-approved input inventory}"
: "${CS406_E3_OUTPUT:?Set the declared run directory}"
: "${CS406_E3_ROOT_SEED:?Set the declared root seed}"
case "$CS406_E3_ROOT_SEED" in 1001|2001|3001|4001) ;; *) echo 'Unreviewed E3 seed' >&2; exit 2 ;; esac
if [[ "${SLURM_CPUS_PER_TASK:-}" != 4 || "${SLURM_MEM_PER_NODE:-}" != 16384 ]]; then
    echo 'Expected 4 CPUs and 16384 MiB allocation' >&2; exit 2
fi
if [[ $# != 0 ]] && [[ $# != 2 || "${1:-}" != --resume ]]; then
    echo 'Usage: e3_head.sbatch [--resume /same/run/last.pt]' >&2; exit 2
fi
export PYTHONHASHSEED="$CS406_E3_ROOT_SEED"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CS406_REPO"
"$CS406_PYTHON" - "$@" <<'PY'
import os,sys
from pathlib import Path
from src.e3.config import read_json,verify_payload
from src.e3.runtime import inspect_inputs
r=read_json(Path(os.environ['CS406_E3_REQUEST']))
i=inspect_inputs(r)
if i['operation'] not in {'pilot_head','final_head'}:
    raise SystemExit('A head operation declaration is required')
d=read_json(verify_payload(r['scientific_declaration']))
if str(d['root_seed'])!=os.environ['PYTHONHASHSEED']:
    raise SystemExit('Root seed differs from immutable declaration')
if len(sys.argv)>1:
    expected=Path(os.environ['CS406_E3_OUTPUT']).resolve()/'last.pt'
    if Path(sys.argv[2]).resolve()!=expected or not expected.is_file():
        raise SystemExit('Resume requires existing same-run last.pt')
PY
exec srun --ntasks=1 --unbuffered "$CS406_PYTHON" -m src.e3 run-head --request "$CS406_E3_REQUEST" --inventory "$CS406_E3_INVENTORY" --output "$CS406_E3_OUTPUT" "$@"

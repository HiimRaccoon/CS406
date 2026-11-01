#!/usr/bin/env bash
#SBATCH --job-name=e1-one-run
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

# One immutable declaration, one process, one run. This file never submits jobs.
set -euo pipefail
: "${CS406_REPO:?Set the clean accepted E1 checkout}"
: "${CS406_PYTHON:?Set the reviewed environment Python executable}"
: "${CS406_CONFIG:?Set one persisted E1 run declaration}"
: "${CS406_E1_SEED:?Set the official seed from the declaration}"
case "$CS406_E1_SEED" in
    1001|2001|3001|4001) ;;
    *) echo 'Unreviewed E1 seed' >&2; exit 2 ;;
esac
if [[ "${SLURM_CPUS_PER_TASK:-}" != 4 || "${SLURM_MEM_PER_NODE:-}" != 16384 ]]; then
    echo 'Expected 4 CPUs and 16384 MiB Slurm allocation' >&2
    exit 2
fi
if [[ $# != 0 ]] && [[ $# != 2 || "${1:-}" != --resume ]]; then
    echo 'Usage: e1_train.sbatch [--resume /same/declared/run/last.pt]' >&2
    exit 2
fi

# Set all four BEFORE the first Python interpreter starts, including validation.
export PYTHONHASHSEED="$CS406_E1_SEED"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CS406_REPO"
"$CS406_PYTHON" - "$@" <<'PY'
import os
import sys
from pathlib import Path
from src.e1.config import read_json, validate_run

config = validate_run(read_json(Path(os.environ['CS406_CONFIG'])))
if str(config['seed']) != os.environ['PYTHONHASHSEED']:
    raise SystemExit('Seed must match the immutable run declaration')
if len(sys.argv) > 1:
    resume = Path(sys.argv[2]).resolve()
    expected = (Path(config['paths']['runs_dir']) / config['run_id'] / 'last.pt').resolve()
    if resume != expected or not resume.is_file():
        raise SystemExit('Resume requires this same declared run and existing last.pt')
PY

# The runner independently enforces actual allocation, G2, stage approval,
# current data hashes, accepted source/runtime, and checkpoint identity.
exec srun --ntasks=1 --unbuffered "$CS406_PYTHON" -m src.e1.train --config "$CS406_CONFIG" "$@"

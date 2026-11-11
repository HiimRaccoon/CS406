#!/usr/bin/env bash
#SBATCH --job-name=e3-extract
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
: "${CS406_E3_OUTPUT:?Set the declared output directory}"
: "${CS406_E3_ROOT_SEED:?Set the declared root seed}"
: "${CS406_E3_OPERATION:?Set accept-source or extract-cache}"
case "$CS406_E3_ROOT_SEED" in 1001|2001|3001|4001) ;; *) echo 'Unreviewed E3 seed' >&2; exit 2 ;; esac
case "$CS406_E3_OPERATION" in accept-source|extract-cache) ;; *) echo 'Wrong extraction operation' >&2; exit 2 ;; esac
if [[ $# != 0 || "${SLURM_CPUS_PER_TASK:-}" != 4 || "${SLURM_MEM_PER_NODE:-}" != 16384 ]]; then
    echo 'Expected no extra arguments, 4 CPUs and 16384 MiB allocation' >&2; exit 2
fi
export PYTHONHASHSEED="$CS406_E3_ROOT_SEED"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CS406_REPO"
"$CS406_PYTHON" - <<'PY'
import os
from pathlib import Path
from src.e3.config import read_json,verify_payload
from src.e3.runtime import inspect_inputs
r=read_json(Path(os.environ['CS406_E3_REQUEST']))
inspection=inspect_inputs(r)
d=read_json(verify_payload(r['scientific_declaration']))
expected=1001 if inspection['operation']=='bounded_acceptance' else d['source_seed']
if str(expected)!=os.environ['PYTHONHASHSEED']:
    raise SystemExit('Root seed differs from immutable declaration')
command='accept-source' if inspection['operation']=='bounded_acceptance' else 'extract-cache'
if command!=os.environ['CS406_E3_OPERATION']:
    raise SystemExit('Operation differs from immutable declaration')
PY
args=("$CS406_E3_OPERATION" --request "$CS406_E3_REQUEST" --inventory "$CS406_E3_INVENTORY" --output "$CS406_E3_OUTPUT")
if [[ "$CS406_E3_OPERATION" == accept-source ]]; then args+=(--gpu); fi
exec srun --ntasks=1 --unbuffered "$CS406_PYTHON" -m src.e3 "${args[@]}"

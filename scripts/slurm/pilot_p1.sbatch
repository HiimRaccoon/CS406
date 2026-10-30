#!/usr/bin/env bash
#SBATCH --job-name=cs406-pilot-p1
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@180
#SBATCH --output=logs/%x-%j.out
# USR1 reaches the srun step. Finish the epoch if possible; otherwise use the last completed epoch.
set -euo pipefail
: "${CS406_REPO:?Set the absolute repository path}"
: "${CS406_PYTHON:?Set the absolute venv Python executable}"
: "${CS406_CONFIG:?Set the reviewed bounded P1 config path}"
: "${SLURM_CPUS_PER_TASK:?Pass --cpus-per-task}"
: "${SLURM_MEM_PER_NODE:?Pass --mem, not --mem-per-cpu}"
(( SLURM_CPUS_PER_TASK <= 16 && SLURM_MEM_PER_NODE <= 73728 )) || {
  echo 'Allocation exceeds documented UIT CPU/RAM limits' >&2; exit 2;
}
cd "$CS406_REPO"
export PYTHONUNBUFFERED=1
srun --ntasks=1 --unbuffered "$CS406_PYTHON" -m src.pilot.p1_train --config "$CS406_CONFIG" "$@"

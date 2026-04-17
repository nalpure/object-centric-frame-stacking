#!/bin/bash

#SBATCH --job-name=slot_finish
#SBATCH --output=out/%x.log
#SBATCH --time=960
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2
#SBATCH --mail-user=david.drexel@uibk.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=n037

module purge
module load cuda/12.6.2-gcc-13.3.0-n5c5eu7
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

uv run train.py dis_dyn.toml -n slot_dyn_dis -d data/4b_4c_fast.npz -b slot_dyn_fast_0 -e 50
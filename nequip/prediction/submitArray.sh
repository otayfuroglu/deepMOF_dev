#!/bin/bash -l
#SBATCH --job-name="nequip"
#SBATCH --account=s1167
#SBATCH --partition=normal
##SBATCH --partition=debug
#SBATCH --time=24:00:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu

#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --array=0-9


# runfile="$(printf 'run-%0.5i.sh' $id)"
# rundir="$(printf 'run-%0.5i' $id)"



export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load daint-gpu
module load PyTorch

idx=$SLURM_ARRAY_TASK_ID

python calcFreeEwithNequip.py -extxyz_path vasp_opt_lowest_10_polymeric_24atoms.extxyz -idx $idx 
exit

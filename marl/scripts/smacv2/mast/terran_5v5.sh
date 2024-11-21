#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH -o ../_out/%j.sbatch.%N.out
#SBATCH -e ../_err/%j.sbatch.%N.err
#================================================

GRES="gpu:1"
mkdir -p ../_log/$SLURM_JOB_ID
SLURM_JOB_PARTITION="gpu1"
cpus_per_task=20

# print sbatch job 
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

env="smacv2"
algo="mast" 
exp_name="MAST"
group_name="MAST64"
map_name=terran_5_vs_5

for seed in 0 1 2 3 4; do
    srun --partition=$SLURM_JOB_PARTITION \
        --gres=$GRES \
        --cpus-per-task=$cpus_per_task \
        -o ../_log/%j/%N.out \
        -e ../_log/%j/%N.err \
        python ../../../train.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp_name} \
        --map_name ${map_name} --seed ${seed} --use_wandb --group_name ${group_name} --log_dir "../../../examples/results";
done
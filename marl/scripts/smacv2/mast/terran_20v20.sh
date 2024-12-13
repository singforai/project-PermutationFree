#!/bin/bash


GRES="gpu:1"
mkdir -p ../_log/$SLURM_JOB_ID
SLURM_JOB_PARTITION="gpu1"
cpus_per_task=20

# print sbatch job 
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

env="smacv2"
algo="mast"
exp_name="PFT"
group_name="PFT_curriculum"
map_name=terran_20_vs_20
# "
for seed in 42; do
    srun --partition=$SLURM_JOB_PARTITION \
        --gres=$GRES \
        --cpus-per-task=$cpus_per_task \
        -o ../_log/%j/%N.out \
        -e ../_log/%j/%N.err \
        python ../../../train.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp_name} --model_dir "../../../models/terran_5v5/PFT/3124"\
        --map_name ${map_name} --seed ${seed} --use_wandb --group_name ${group_name} --log_dir "../../../examples/results";
done
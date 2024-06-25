#!/bin/bash

# Arrays of hyperparameters to grid search over
lrs=("5e-3")
dropouts=("0.0")
gt_choose_list=("random" "decide")

# Nested loops to iterate over all combinations of hyperparameters
for lr in "${lrs[@]}"; do
    for dropout in "${dropouts[@]}"; do
        for gt_choose in "${gt_choose_list[@]}"; do

            # Generate a unique output file for each job to avoid overwriting
            output_file="/home/mila/h/haolun.wu/projects/Disk-SNAKE/exp_out/date-set-order-N_100_lr${lr}_dropout${dropout}_gt_${gt_choose}.out"

            # Launch a separate job for each hyperparameter combination
            sbatch <<EOL
#!/bin/bash
#SBATCH --output=${output_file}
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --partition=unkillable

source /home/mila/h/haolun.wu/projects/Disk-SNAKE/venv/bin/activate
module load python/3.10

nvidia-smi

# Run your script with the current hyperparameter combination
python3 /home/mila/h/haolun.wu/projects/Disk-SNAKE/train.py \
    --dataset=date-set-order-N \
    --num_data=100 \
    --epochs=5000 \
    --wandb=1 \
    --lr=${lr} \
    --dropout=${dropout} \
    --gt_choose=${gt_choose}

deactivate
EOL

        done
    done
done

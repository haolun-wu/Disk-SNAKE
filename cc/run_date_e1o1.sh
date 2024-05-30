#!/bin/bash
#SBATCH --output=/home/mila/h/haolun.wu/projects/Disk-SNAKE/exp_out/date-set-order-Y.out
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000M

source /home/mila/h/haolun.wu/projects/Disk-SNAKE/venv/bin/activate
module load python/3.10
# module load cuda

nvidia-smi

python3 /home/mila/h/haolun.wu/projects/Disk-SNAKE/train.py --dataset=date-set-order-Y --epochs=100 --wandb=1 --num_data=10000

deactivate

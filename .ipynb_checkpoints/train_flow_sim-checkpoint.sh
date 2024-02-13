#! /bin/sh
#
#SBATCH --job-name=train_teacher
#
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --time=30:00:00
#SBATCH -N 1
#SBATCH -G 1

python3 flows/flow_sim_8.py
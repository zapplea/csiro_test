#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="che313"
#SBATCH --time=140:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
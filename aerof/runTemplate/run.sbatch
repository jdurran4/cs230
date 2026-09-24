#!/bin/bash
#SBATCH --job-name=CASE
#SBATCH --time=10:00:00
#SBATCH --ntasks=48
##SBATCH --partition=cfarhat
#SBATCH --ntasks-per-node=12
#SBATCH --mem=16G

./run.sh

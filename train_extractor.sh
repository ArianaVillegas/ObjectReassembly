#!/bin/bash
#SBATCH --job-name=vn_dgcnn     # nombre del job
#SBATCH --nodes=1                # cantidad de nodosa89p
#SBATCH --ntasks=1               # cantidad de tareas
#SBATCH --cpus-per-task=1        # cpu-cores por task 
#SBATCH --mem=5G                # memoria total por nodo
#SBATCH --gres=gpu:1             # numero de gpus por nodo
#SBATCH --exclude=g001

module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate torch

python train_extractor.py --cfg_file=config/extractor/train_extractor_dgcnn.yml

conda deactivate
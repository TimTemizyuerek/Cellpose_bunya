#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --job-name=cellpose_training
#SBATCH --time=120:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=a_senv_ege
#SBATCH -o cellpose_training.output
#SBATCH -e cellpose_training.error

mkdir -p $TMPDIR/training_images
rsync -az /scratch/user/uqttemiz/training_images/ $TMPDIR/training_images/

module load miniforge/26.1.0-0
source $ROOTMINIFORGE/etc/profile.d/conda.sh
conda activate cellpose-env

srun python /scratch/user/uqttemiz/bunya_cellpose_training.py

# Copy the trained model out of $TMPDIR before it's deleted
mkdir -p /scratch/user/uqttemiz/trained_models
cp -r $TMPDIR/training_images/ /scratch/user/uqttemiz/trained_models/
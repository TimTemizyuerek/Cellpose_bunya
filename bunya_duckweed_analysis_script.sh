#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --job-name=cellpose_analysis
#SBATCH --time=120:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=a_senv_ege
#SBATCH -o cellpose_analysis.output
#SBATCH -e cellpose_analysis.error

mkdir -p $TMPDIR/analysis_images
rsync -az /scratch/user/uqttemiz/analysis_images/ $TMPDIR/analysis_images/

module load miniforge/26.1.0-0
source $ROOTMINIFORGE/etc/profile.d/conda.sh
conda activate cellpose-env

srun python /scratch/user/uqttemiz/bunya_duckweed_individual_fonds.py

# check for early script failure
if [ $? -ne 0 ]; then
    echo "Analysis script failed"
    exit 1
fi

# Copy the images out of the temoporary directory
mkdir -p /scratch/user/uqttemiz/analysis_output
rsync -a $TMPDIR/analysis_output/ /scratch/user/uqttemiz/analysis_output/
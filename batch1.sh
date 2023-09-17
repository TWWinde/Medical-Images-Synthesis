#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=medical
#SBATCH --output=medical_no_3d_noise%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodes=1


# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
pyenv activate venv
module load cuda/11.3

# Run your python code
# For single GPU use this


CUDA_VISIBLE_DEVICES=0 python train.py --name medicals_no_3d_noise --dataset_mode medicals --gpu_ids 0 \
--dataroot /misc/data/private/autoPET  \
--batch_size 4 --model_supervision 0  \
--Du_patch_size 32 --netDu wavelet  \
--netG 0 --channels_G 64 \
--num_epochs 500 \
--no_3dnoise

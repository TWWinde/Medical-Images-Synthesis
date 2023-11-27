#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=medical
#SBATCH --output=medical%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
# SBATCH --nodes=1
# SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
pyenv activate moose_env
module load cuda
# module load cuda/11.3

# Run your python code
# For single GPU use this
# CUDA_VISIBLE_DEVICES=0 python /no_backups/s1449/OASIS/dataloaders/get_2D_images.py
#--name USIS_cityscapes --dataset_mode cityscapes --gpu_ids 0 \
#--dataroot /data/public/cityscapes  \
#--batch_size 2 --model_supervision 0  \
#--Du_patch_size 64 --netDu wavelet  \
#--netG 0 --channels_G 64 \
#--num_epochs 500

#CUDA_VISIBLE_DEVICES=0 python train.py --name medicals --dataset_mode medicals --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/data_nnunet  \
#--batch_size 4 --model_supervision 0 --add_mask \
#--Du_patch_size 32 --netDu wavelet  \
#--netG 0 --channels_G 64 \
#--num_epochs 500

CUDA_VISIBLE_DEVICES=0 python /misc/no_backups/s1449/Medical-Images-Synthesis/utils/miou_folder/moose_segment.py


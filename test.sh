#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=medical
#SBATCH --output=MRI%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
# SBATCH --nodes=1
#SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
pyenv activate venv
module load cuda



python test.py --name medicals --dataset_mode medicals --gpu_ids 0 \
--dataroot /misc/data/private/autoPET/data_nnunet --batch_size 20 --model_supervision 0  \


#experiments_1
CUDA_VISIBLE_DEVICES=0 python test.py --name Oasis_MRI --dataset_mode medicals --gpu_ids 0 \
--dataroot /misc/data/private/autoPET/CT_MR  \
--batch_size 20 --model_supervision 0 --add_mask \
--Du_patch_size 32 --netDu wavelet  \
--netG 0 --channels_G 64

#experiments_2
#CUDA_VISIBLE_DEVICES=0 python train.py --name Wavelet_MRI --dataset_mode medicals --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR  \
#--batch_size 4 --model_supervision 0 --add_mask \
#--Du_patch_size 32 --netDu wavelet  \
#--netG 9 --channels_G 16
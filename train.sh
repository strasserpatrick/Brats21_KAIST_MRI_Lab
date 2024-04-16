#!/bin/bash
source ~/env_file

eval "$(conda shell.bash hook)"
conda activate extended-nnunet 


echo "starting training with fold $fold"
nnUNet_train 3d_fullres nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm 500 $fold --npz

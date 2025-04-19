#!/bin/bash
#SBATCH -J neuralcf-train
#SBATCH --output=neuralcf-train.out
#SBATCH -N 1                            
#SBATCH --gres=gpu:A100:1  
#SBATCH --mem=128G                 
#SBATCH -t 15:00:00                       
#SBATCH --mail-type=BEGIN,END,FAIL              
#SBATCH --mail-user=nvasan7@gatech.edu       

# Load any necessary modules
module anaconda3

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate DLT-FP-2
module load cuda/12.1.1

# Run your training script
srun python train.py
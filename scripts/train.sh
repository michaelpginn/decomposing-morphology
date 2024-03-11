#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=train.%j.out      # Output file name
#SBATCH --error=train.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/decomposing-morphology/src"

for seed in 1 2 3 4 5
do
  python3 train.py --seed $seed --features_path ../features_v1.csv
done
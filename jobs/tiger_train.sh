#!/bin/bash
#SBATCH --job-name=tiger_train
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/tiger_train_%j.out
#SBATCH --error=logs/tiger_train_%j.err

MODEL_SIZE_INPUT=$1  # Pass L or XXL
DATASET=$2           # Pass beauty, toys, or sports

if [[ -z "$MODEL_SIZE_INPUT" || -z "$DATASET" ]]; then
    echo "Usage: sbatch tiger_train.sh [L|XXL] [beauty|toys|sports]"
    exit 1
fi

# Standardise case handling just like your other scripts
MODEL_SIZE=$(echo "$MODEL_SIZE_INPUT" | tr '[:lower:]' '[:upper:]')

module purge
module load 2025
module load Anaconda3/2025.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RecSys

export OMP_NUM_THREADS=8
cd $HOME/GRID

BASE_PROJECT=/projects/prjs2120/groups/group_08
BASE_SCRATCH=/scratch-shared/$USER

# Point semantic_id_path to where rkmeans_inference saved the outputs on scratch
SEMANTIC_ID_PATH=$BASE_SCRATCH/results/sid_rkmeans/${MODEL_SIZE}/${DATASET}/rkmeans_inference/pickle/merged_predictions_tensor.pt

# Run the final generative recommendation training completely on scratch
python -m src.train experiment=tiger_train_flat \
    data_dir=$BASE_PROJECT/data/amazon_data/$DATASET \
    semantic_id_path=$SEMANTIC_ID_PATH \
    num_hierarchies=4 \
    hydra.run.dir=$BASE_SCRATCH/results/tiger/${MODEL_SIZE}/${DATASET}_rkmeans
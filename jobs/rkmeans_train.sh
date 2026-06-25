#!/bin/bash
#SBATCH --job-name=train_sids
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

MODEL_SIZE=$1
DATASET=$2

if [[ -z "$MODEL_SIZE" || -z "$DATASET" ]]; then
    echo "Usage: sbatch rkmeans_train.sh [L|XXL] [beauty|toys|sports]"
    exit 1
fi

if [ "$MODEL_SIZE" = "L" ]; then
    DIM=1024
elif [ "$MODEL_SIZE" = "XXL" ]; then
    DIM=4096
else
    echo "Invalid model size. Choose L or XXL."
    exit 1
fi

module purge
module load 2025
module load Anaconda3/2025.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RecSys

cd $HOME/GRID
export OMP_NUM_THREADS=8
BASE_PROJECT=/projects/prjs2120/groups/group_08
BASE_SCRATCH=/scratch-shared/$USER

# Point directly to your scratch embeddings
EMBED_PATH=$BASE_SCRATCH/embeddings/${MODEL_SIZE}/${DATASET}/pickle/merged_predictions_tensor.pt

# Ensure the log/metadata structure exists in the project folder (lightweight text files)
mkdir -p $BASE_PROJECT/results/sid_rkmeans/${MODEL_SIZE}/${DATASET}/rkmeans_train/metadata

# Force the heavy checkpoint outputs into your scratch playground
python -m src.train experiment=rkmeans_train_flat \
    data_dir=$BASE_PROJECT/data/amazon_data/$DATASET \
    embedding_path=$EMBED_PATH \
    embedding_dim=$DIM \
    num_hierarchies=3 \
    codebook_width=256 \
    hydra.run.dir=$BASE_SCRATCH/results/sid_rkmeans/${MODEL_SIZE}/${DATASET}/rkmeans_train
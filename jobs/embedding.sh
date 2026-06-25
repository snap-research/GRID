#!/bin/bash
#SBATCH --job-name=embeds
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=jobs/outputs/%x-%j.out
#SBATCH --error=jobs/outputs/%x-%j.err

MODEL_SIZE_INPUT=$1  # Pass L or XXL
DATASET=$2           # Pass beauty, toys, or sports

if [[ -z "$MODEL_SIZE_INPUT" || -z "$DATASET" ]]; then
    echo "Usage: sbatch embedding.sh [L|XXL] [beauty|toys|sports]"
    exit 1
fi

MODEL_SIZE=$(echo "$MODEL_SIZE_INPUT" | tr '[:lower:]' '[:upper:]')

if [[ "$MODEL_SIZE" == "L" || "$MODEL_SIZE" == "LARGE" ]]; then
    HF_MODEL="large"
elif [[ "$MODEL_SIZE" == "XXL" ]]; then
    HF_MODEL="xxl"
else
    echo "Invalid model size: $MODEL_SIZE_INPUT. Choose L or XXL."
    exit 1
fi

cd $HOME/GRID

module purge
module load 2025
module load Anaconda3/2025.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RecSys

# --- FORCE ALL STORAGE & TEMP WRITES TO SCRATCH ---
export HF_HOME=/scratch-shared/$USER/hf_cache
export TMPDIR=/scratch-shared/$USER/tmp
mkdir -p $HF_HOME
mkdir -p $TMPDIR

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export OMP_NUM_THREADS=8

BASE=/projects/prjs2120/groups/group_08

python -m src.inference \
    experiment=sem_embeds_inference_flat \
    data_dir=$BASE/data/amazon_data/$DATASET \
    hydra.run.dir=/scratch-shared/$USER/embeddings/${MODEL_SIZE}/$DATASET \
    embedding_model=google/flan-t5-${HF_MODEL}
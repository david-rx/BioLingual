#!/bin/bash
#SBATCH --comment clap
#SBATCH --partition=g40423
#SBATCH --job-name=mclap
#SBATCH --nodes 3
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=6
#SBATCH --exclusive
#SBATCH --output=%x_%j.out

# module load openmpi
# module load cuda/11.7
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

# sent to sub script
# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=12802
# export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export WORLD_SIZE=2
export RANK=0

torchrun --nproc_per_node=2 -m src.laion_clap.training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --datasetpath="./" \
    --precision="fp32" \
    --batch-size=170 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=2 \
    --use-bn-sync \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --warmup 3200 \
    --report-to "wandb" \
    --datasetnames "animals" \
    --datasetinfos "eval" \
    --top-k-checkpoint-select-dataset="AnimalCLAP-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs logs \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained-audio HTSAT.ckpt
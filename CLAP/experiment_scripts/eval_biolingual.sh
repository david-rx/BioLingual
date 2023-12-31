python -m src.laion_clap.training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --datasetpath="." \
    --precision="fp32" \
    --batch-size=64 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=56 \
    --workers=1 \
    --use-bn-sync \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --warmup 3200 \
    --datasetnames "animals_webdataset" \
    --datasetinfos "train" \
    --top-k-checkpoint-select-dataset="AnimalCLAP-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs 'logs' \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --prefetch-factor 2 \
    --resume "../../beans/BioLingual.pt"
    # --resume "630k-audioset-best.pt"
    # --resume "../../beans/BioLingual.pt"
    # --resume "630k-audioset-best.pt"    
    # --resume "../../beans/BioLingual.pt"
    # --device "mps"
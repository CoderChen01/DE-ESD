#!/bin/zsh

em_extractor_names=(
    bert-base
    bert-large
    roberta-base
    roberta-large
    deberta-base
    deberta-large
    pegasus-base
    pegasus-large
    bart-base
    bart-large
    t5-base
    t5-large
)
dataset_name=$1

for em_extractor_name in $em_extractor_names
do
    python scripts/de_esd_training.py  \
        --dataset_path ../data/${dataset_name}/dataset \
        --output_dir ../data/${dataset_name}/checkpoints/${em_extractor_name}/ \
        --logging_steps 100 \
        --num_layers 1 \
        --set_scorer_dropout 0.3 \
        --do_train \
        --load_best_model_at_end \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --save_steps 200 \
        --metric_for_best_model eval_mse \
        --greater_is_better False \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32  \
        --num_train_epochs 6 \
        --em_extractor_name $em_extractor_name \
        --logging_dir ../data/${dataset_name}/checkpoints/${em_extractor_name}/
done
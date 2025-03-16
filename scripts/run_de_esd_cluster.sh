#!/bin/zsh

# em_extractor_names=(bert-base bert-large roberta-base distilbert-base albert-base deberta-base bart-base bart-large t5-small t5-base t5-large gpt2 xlnet-base xlnet-large opt longformer-base pegasus-large)
# em_extractor_names=(t5-base pegasus-large)
em_extractor_names=(
    # bert-base
    # bert-large
    # roberta-base
    # roberta-large
    # deberta-base
    # deberta-large
    # pegasus-base
    # pegasus-large
    # bart-base
    # bart-large
    t5-base
    t5-large
)

dataset_name=$1
ctx_size=$2

echo $(pwd)

for em_extractor_name in $em_extractor_names
do
    python scripts/synset_cluster.py  \
           -c de-esd \
           -t 0.5 0.9 0.1 \
           -d ../data/${dataset_name}/dataset/all_test_data/test_data_ctx-num-${ctx_size}.jsonl \
           -s ../data/${dataset_name}/cluster-results/de-esd-prf1/${em_extractor_name} \
           -g ../data/${dataset_name}/dataset/test_data_gt.csv \
           -m ../data/${dataset_name}/checkpoints/${em_extractor_name}/
done


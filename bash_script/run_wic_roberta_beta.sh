export TASK_NAME=superglue
export DATASET_NAME=wic
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=1e-2
dropout=0.1
psl=8
epoch=60
tau=8
alpha=1e-3


    python3 run.py \
    --model_name_or_path /data/zhanghy/P-tuning-v2/local_models/roberta-large \
    --task_name $TASK_NAME \
    --dataset_name $DATASET_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --pre_seq_len $psl \
    --output_dir checkpoints/$DATASET_NAME-roberta/tau$tau=alpha$alpha=bs$bs=psl$psl \
    --overwrite_output_dir \
    --hidden_dropout_prob $dropout \
    --seed 11 \
    --save_strategy no \
    --evaluation_strategy epoch \
    --selective_prefix \
    --tau $tau \
    --overwrite_cache \
    --alpha $alpha

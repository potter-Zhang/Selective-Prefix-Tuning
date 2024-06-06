export TASK_NAME=superglue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0

bs=16
lr=1e-2
dropout=0.1
psl=8
epoch=50
tau=1
alpha=2e-4


python3 run.py \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --dataset_name $DATASET_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --pre_seq_len $psl \
    --output_dir checkpoints/$DATASET_NAME-bert-base/ \
    --overwrite_output_dir \
    --hidden_dropout_prob $dropout \
    --seed 44 \
    --save_strategy no \
    --evaluation_strategy epoch \
    --selective_prefix \
    --tau $tau \
    --alpha $alpha \
    --overwrite_cache \
    --gradient_accumulation_step 2

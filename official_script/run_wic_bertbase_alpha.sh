export TASK_NAME=superglue
export DATASET_NAME=wic
export CUDA_VISIBLE_DEVICES=0

bs=16
lr=7e-3
dropout=0.1
psl=20
epoch=80
tau=8
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
  --output_dir checkpoints/$DATASET_NAME-bertbase \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --template_id 1 \
  --selective_prefix \
  --overwrite_cache \
  --tau $tau \
  --alpha $alpha
